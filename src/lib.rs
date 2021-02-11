#![cfg_attr(feature = "bench", feature(test))]
#![feature(linked_list_cursors)]

#[cfg(feature = "bench")]
extern crate test;

use std::fs::File;
use std::io;
use std::io::Write;
use fst::{IntoStreamer, Map, MapBuilder};
use memmap::Mmap;
use membuffer::{MemBufferWriter,MemBufferReader};
use std::borrow::Cow;
use regex::Regex;
use serde::{Serialize,Deserialize};
use levenshtein_automata::{Distance, LevenshteinAutomatonBuilder, DFA,self};
use bincode;
use std::time;

struct DocumentInfo {
    doc_ptr: u64,
    doc_freq: i32,
}

struct OrderedCollector<'a,X: Serialize+Deserialize<'a>> {
    traversal: std::collections::LinkedList<SearchHit<'a,X>>,
}

impl<'a,X: Serialize+Deserialize<'a>> OrderedCollector<'a,X> {
    pub fn new() -> OrderedCollector<'a,X> {
        OrderedCollector {
            traversal: std::collections::LinkedList::new(),
        }
    }

    pub fn finalize(self) -> std::collections::LinkedList<SearchHit<'a,X>> {
        self.traversal
    }

    pub fn insert_all(&mut self,idx: &'a [DocumentInfo], word_index: i32, word_pos: &[&'a [u32]], buffer: &'a [u8]) {
        for x in 0..idx.len() {
            println!("{}",idx[x].doc_ptr);
            let information = MemBufferReader::new(&buffer[idx[x].doc_ptr as usize..]).unwrap();
            let search = SearchHit {
                doc_ptr: idx[x].doc_ptr,
                metadata: information.load_serde_entry::<X>(0).unwrap(),
                body: information.load_entry(1).unwrap(),
                doc_score: 0.0,
                matched_words: vec![word_index],
                positions: vec![word_pos[x]]
            };
            self.traversal.push_back(search);
        }
    }
    
    pub fn add_array(&mut self, idx: &'a [DocumentInfo], word_index: i32, word_pos: Vec<&'a [u32]>, buffer: &'a [u8]) {
        if self.traversal.len() == 0 {
            self.insert_all(idx, word_index, &word_pos[..], buffer);
            return ();
        }

        let mut cursor_front = self.traversal.cursor_front_mut();
        let mut index_other = 0;
        loop {
            if let Some(curr_doc_info) = cursor_front.current() {
                let curr_ptr = curr_doc_info.doc_ptr;
                if curr_ptr < idx[index_other].doc_ptr {
                    cursor_front.move_next();
                    continue;
                }
                else if curr_ptr == idx[index_other].doc_ptr {
                    curr_doc_info.matched_words.push(word_index);
                    curr_doc_info.positions.push(word_pos[index_other]);
                }
                else {
                    let information = MemBufferReader::new(&buffer[idx[index_other].doc_ptr as usize..]).unwrap();
                    let x = SearchHit {
                        doc_ptr: idx[index_other].doc_ptr,
                        metadata: information.load_serde_entry::<X>(0).unwrap(),
                        body: information.load_entry(1).unwrap(),
                        doc_score: 0.0,
                        matched_words: vec![word_index],
                        positions: vec![word_pos[index_other]]
                    };
                    cursor_front.insert_before(x);
                }

                index_other+=1;
                if index_other >= idx.len() {
                    break;
                }
            }
            else {
                if index_other < idx.len() {
                    println!("Before insert all {}",self.traversal.len());
                    self.insert_all(&idx[index_other..], word_index, &word_pos[index_other..], buffer);
                }
                break;
            }
        }
    }
}



struct DocumentCollection<'a> {
    docs: &'a [DocumentInfo],
}


#[derive(Serialize,Deserialize)]
struct IndexWriterLazy {
    position_list: std::collections::BTreeMap<u64,Vec<u32>>
}

#[derive(Serialize,Deserialize)]
struct IndexInformations {
    index: std::collections::BTreeMap<String,IndexWriterLazy>,
    document_count: u64,
}

struct IndexMemmapWriter {
    values: IndexInformations,
    index_body: Vec<u8>,
    directory: String,
}

impl<'a> membuffer::MemBufferDeserialize<'a,DocumentCollection<'a>> for DocumentCollection<'a> {
    fn from_mem_buffer(pos: &membuffer::Position, mem: &'a [u8]) -> Result<DocumentCollection<'a>,membuffer::MemBufferError> {
        let len = pos.length/std::mem::size_of::<DocumentInfo>() as i32;
        unsafe{ Ok(DocumentCollection{
            docs: std::slice::from_raw_parts(mem[pos.offset as usize..].as_ptr().cast::<DocumentInfo>(),len as usize)} 
            )}
    }
}


impl<'a> membuffer::MemBufferSerialize for DocumentCollection<'a> {
    fn to_mem_buffer<'b>(&'b self, _: &mut membuffer::Position) -> Cow<'b, [u8]> {
        let vals: &'a [u8] = unsafe{ std::slice::from_raw_parts(self.docs.as_ptr().cast::<u8>(),std::mem::size_of::<DocumentInfo>()*self.docs.len() as usize)};
        Cow::Borrowed(vals)
    }

    fn get_mem_buffer_type() -> i32 {
        10
    }
}

struct Document<'a,X: Serialize+Deserialize<'a>> {
    metadata: X,
    body: &'a str,
}


impl IndexMemmapWriter {
    pub fn new(directory: &str) -> IndexMemmapWriter {
        if let Ok(data) = std::fs::metadata(directory) {
            if data.is_file() {
                panic!(format!("The path {} exists but is not a directory!",directory));
            }

            let strings = std::fs::read(std::path::Path::new(directory).join("index.bincode")).unwrap();
            return IndexMemmapWriter {
                values: bincode::deserialize(&strings[..]).unwrap(),
                index_body: std::fs::read(std::path::Path::new(directory).join("doc_content.bin")).unwrap(),
                directory: directory.to_string()
            };
        }
        let _ = std::fs::create_dir(directory);

        IndexMemmapWriter {
            values: IndexInformations {
                index: std::collections::BTreeMap::new(),
                document_count: 0,
            },
            index_body: Vec::new(),
            directory: directory.to_string(),
        }
    }

    pub fn term_len(&self) -> usize {
        self.values.index.len()
    }

    pub fn doc_count(&self) -> u64 {
        self.values.document_count
    }

    pub fn commit(&mut self) {
        let string = bincode::serialize(&self.values).unwrap();
        std::fs::write(std::path::Path::new(&self.directory).join("index.bincode"), string).unwrap();
        self.write();
    }

    pub fn add_key(&mut self, key: String, doc_id: u64, index_key_start: u64) {
        if let Some(idx) = self.values.index.get_mut(&key) {
            if let Some(entry) = idx.position_list.get_mut(&doc_id) {
                entry.push(index_key_start as u32);
            }
            else {
                idx.position_list.insert(doc_id,vec![index_key_start as u32]);
            }
        }
        else {
            let mut pos_list = std::collections::BTreeMap::new();
            pos_list.insert(doc_id,vec![index_key_start as u32]);
            self.values.index.insert(key.to_string(),IndexWriterLazy {
                position_list: pos_list
            });
        }
    }

    pub fn add_document<T: Serialize>(&mut self, x: &str, meta: &T) {
        let re = Regex::new(r"\b\S+\b").unwrap();
        let doc_id = self.index_body.len() as u64;
        let mut body = MemBufferWriter::new();
        body.add_serde_entry(meta);
        body.add_entry(x);
        self.index_body.extend_from_slice(&body.finalize());
        for entry in re.find_iter(x) {
            let key = &x[entry.start()..entry.end()];
            if key.len() > 30 {
                continue;
            }
            self.add_key(key.to_lowercase(), doc_id,entry.start() as u64);
        }
        self.values.document_count+=1;
    }

    pub fn write(&mut self) {
        // This is where we'll write our map to.

        let wtr = io::BufWriter::new(File::create(std::path::Path::new(&self.directory).join("finite_state_machine.fst")).unwrap());


        // Create a builder that can be used to insert new key-value pairs.
        let mut build = MapBuilder::new(wtr).unwrap();
        let mut index_data: Vec<u8> = Vec::new();
        println!("Map size {}",self.values.index.len());
        let start = time::SystemTime::now();

        for (key,value) in self.values.index.iter() {
            build.insert(&key, index_data.len() as u64).unwrap();
            let mut documents : Vec<DocumentInfo>  = Vec::new();

            let mut writer2 = MemBufferWriter::new();
            for (x,positions) in value.position_list.iter() {
                documents.push(DocumentInfo {
                    doc_ptr: *x,
                    doc_freq: positions.len() as i32,
                });
                writer2.add_entry(&positions[..]);
            }

            let writer = DocumentCollection {
                docs: &documents
            };

            writer2.add_entry(writer);
            index_data.extend_from_slice(&writer2.finalize());
        }


        // Finish construction of the map and flush its contents to disk.
        build.finish().unwrap();
        println!("Elapsed time: {}",start.elapsed().unwrap().as_millis());
        
        let mut index_data_buf = io::BufWriter::new(File::create(std::path::Path::new(&self.directory).join("term_index.bin")).unwrap());
        index_data_buf.write(&index_data).unwrap();

        let mut body_data = io::BufWriter::new(File::create(std::path::Path::new(&self.directory).join("doc_content.bin")).unwrap());
        body_data.write(&self.index_body).unwrap();
    }
}


pub(crate) struct DFAWrapper(pub DFA);

impl fst::Automaton for DFAWrapper {
    type State = u32;

    fn start(&self) -> Self::State {
        self.0.initial_state()
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match self.0.distance(*state) {
            Distance::Exact(_) => true,
            Distance::AtLeast(_) => false,
        }
    }

    fn can_match(&self, state: &u32) -> bool {
        *state != levenshtein_automata::SINK_STATE
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        self.0.transition(*state, byte)
    }
}


struct IndexMmapReader {
    document_contents: memmap::Mmap,
    document_terms: memmap::Mmap,
    automaton: fst::Map<memmap::Mmap>,
    dist: LevenshteinAutomatonBuilder,
}

struct SearchHit<'a,X: Serialize+Deserialize<'a>> {
    doc_ptr: u64,
    metadata: X,
    body: &'a str,
    doc_score: f32,
    matched_words: Vec<i32>,
    positions: Vec<&'a [u32]>,
}

struct DocumentSearchResult<'a,X: Serialize+Deserialize<'a>> {
    map_of_ids: std::collections::LinkedList<SearchHit<'a,X>>,
    matched_words: Vec<(String, u64)>,
}

impl<'a> IndexMmapReader {
    pub fn new(directory: &str) -> IndexMmapReader {
        let path = std::path::Path::new(directory);
        let file = unsafe{Mmap::map(&File::open(path.join("finite_state_machine.fst")).unwrap()).unwrap()};
        IndexMmapReader {
            document_contents: unsafe{Mmap::map(&File::open(path.join("doc_content.bin")).unwrap()).unwrap()},
            document_terms: unsafe{Mmap::map(&File::open(path.join("term_index.bin")).unwrap()).unwrap()},
            automaton: Map::new(file).unwrap(),
            dist: LevenshteinAutomatonBuilder::new(1, true)
        }
        
    }

    pub fn search<X: Serialize+Deserialize<'a>>(&'a self,string: &str,max_results: usize) -> DocumentSearchResult<'a, X> {
        let dfa = DFAWrapper(self.dist.build_dfa(string));
        let result = self.automaton.search(dfa).into_stream();
        let val : &[u8] = &self.document_terms;
        let stream_result = result.into_str_vec().unwrap();
        

        let mut return_value : DocumentSearchResult<'a,X> = DocumentSearchResult {
            map_of_ids: std::collections::LinkedList::new(),
            matched_words: stream_result
        };

        let mut orderedcoll: OrderedCollector<'a,X> = OrderedCollector::new();
        //58ms
        let mut counter = 0;
        for (_,value) in return_value.matched_words.iter() {
            let reader = MemBufferReader::new(&val[*value as usize..]).unwrap();
            let val: DocumentCollection = reader.load_entry(reader.len()-1).unwrap();
            let mut docs : Vec<&'a [u32]> = Vec::with_capacity(val.docs.len());

            println!("Sending {} docs",val.docs.len());
            for x in 0..val.docs.len() {
                docs.push(reader.load_entry(x).unwrap());
            }

            orderedcoll.add_array(val.docs, counter, docs, &self.document_contents);
            println!("total len: {}",orderedcoll.traversal.len());
            counter+=1;
        }
        return_value.map_of_ids = orderedcoll.finalize();
        return_value
   }
}

#[cfg(test)]
mod tests {
    use super::{IndexMemmapWriter,IndexMmapReader};
    use serde::{Serialize,Deserialize};
    
    #[derive(Serialize,Deserialize)]
    struct DocumentMeta<'a> {
        title: &'a str,
        path: &'a str,
    }

    #[test]
    fn check_simple_search() {
        let mut new_writer = IndexMemmapWriter::new("big_data");
        let hugestring = std::fs::read_to_string("main.txt").unwrap();
        let doc = DocumentMeta {
            title: "The modern Intel System Environment",
            path: "main.txt"
        };
        for x in 0..10 {
            new_writer.add_document(&hugestring, &doc);
        }

        for x in 0..100_000 {
            new_writer.add_document("Hello how are you? I am very well and look forward to relaxing with you like a lot, this could happen a lot more often.", &doc);
        }
        new_writer.commit();

        let reader = IndexMmapReader::new("big_data");
        let results = reader.search::<DocumentMeta>("intel",0);
    }
}


#[cfg(feature="bench")]
mod bench {
    use test::Bencher;
    use super::{IndexMemmapWriter,IndexMmapReader};
    use serde::{Serialize,Deserialize};

    #[derive(Serialize,Deserialize)]
    struct DocumentMeta<'a> {
        title: &'a str,
        path: &'a str,
    }

    #[bench]
    fn check_string_searchin(b: &mut Bencher) {
        let mut value = String::new();
        for x in 0..16192 {
            value.push('b');
        }
        value+="nice";
        let mut counter = 0;

        b.iter(|| {
            if let Some(result) = value.find("nice") {
                counter+=1;
            }
        });
    }



    #[bench]
    fn check_reading_own(b: &mut Bencher) {
        let mut new_writer = IndexMemmapWriter::new("big_data");
        let hugestring = std::fs::read_to_string("main.txt").unwrap();
        let hugestring1 = std::fs::read_to_string("second.txt").unwrap();
        let hugestring2 = std::fs::read_to_string("third.txt").unwrap();
        let doc = DocumentMeta {
            title: "The modern Intel System Environment",
            path: "main.txt"
        };
        new_writer.add_document(&hugestring,&doc);
        new_writer.add_document(&hugestring,&doc);
        new_writer.add_document(&hugestring1,&doc);
        new_writer.add_document(&hugestring2,&doc);
        for x in 0..10 {
            new_writer.add_document(&hugestring, &doc);
        }

        for x in 0..1_000 {
            new_writer.add_document("Hello how are you? I am very well and look forward to relaxing with you like a lot, this could happen a lot more often.", &doc);
        }
        new_writer.commit();

        b.iter(|| {
            let reader = IndexMmapReader::new("big_data");
            let result = reader.search::<DocumentMeta>("and",0);
            assert_eq!(result.map_of_ids.len(),1_014);
            let mut score = 0.0;
            for x in result.map_of_ids.iter() {
                score+=x.doc_score;
            }
        });
    }

}
