#![cfg_attr(feature = "bench", feature(test))]

#[cfg(feature = "bench")]
extern crate test;

use std::fs::File;
use std::io::Write;
use fst::{IntoStreamer, MapBuilder,Automaton};
use memmap::Mmap;
use membuffer::{MemBufferWriter,MemBufferReader};
use std::borrow::Cow;
use regex::Regex;
use serde::{Serialize,Deserialize};
use levenshtein_automata::{Distance, LevenshteinAutomatonBuilder, DFA,self};
use bincode;
use fst::Streamer;
use walkdir::WalkDir;

pub struct Document<'a, X: Serialize+Deserialize<'a>> {
    metadata: &'a X,
    indexable_parts: std::collections::HashMap<i32, &'a str>, 
}

#[macro_use]
extern crate lazy_static;

impl<'a,X: Serialize + Deserialize<'a>> Document<'a,X> {
    pub fn new(metadata: &'a X) -> Document<'a,X> {
        Document {
            metadata,
            indexable_parts: std::collections::HashMap::new(),
        }
    }

    pub fn add_field<T: Into<i32>>(&mut self,index: T, index_content: &'a str) {
        let index_numerical = index.into();
        if let Some(_) = self.indexable_parts.get(&index_numerical) {
            panic!("There is already a field associated with the index");
        }
        self.indexable_parts.insert(index_numerical,index_content);
    }
}

pub trait Filesystem {
    fn load_file(&self, path: &str) -> Result<&[u8],std::io::Error>;
    fn write_file(&mut self, path: &str, slice: &[u8]) -> Result<(),std::io::Error>;
}

pub struct DocumentTermInfo {
    doc_ptr: u64,
    doc_freq: f32,
}

pub struct DocumentTermCollection<'a> {
    docs: &'a [DocumentTermInfo],
}

impl<'a> membuffer::MemBufferDeserialize<'a,DocumentTermCollection<'a>> for DocumentTermCollection<'a> {
    fn from_mem_buffer(pos: &membuffer::Position, mem: &'a [u8]) -> Result<DocumentTermCollection<'a>,membuffer::MemBufferError> {
        let len = pos.length/std::mem::size_of::<DocumentTermInfo>() as i32;
        unsafe{ Ok(DocumentTermCollection{
            docs: std::slice::from_raw_parts(mem[pos.offset as usize..].as_ptr().cast::<DocumentTermInfo>(),len as usize)} 
            )}
    }
}


impl<'a> membuffer::MemBufferSerialize for DocumentTermCollection<'a> {
    fn to_mem_buffer<'b>(&'b self, _: &mut membuffer::Position) -> Cow<'b, [u8]> {
        let vals: &'a [u8] = unsafe{ std::slice::from_raw_parts(self.docs.as_ptr().cast::<u8>(),std::mem::size_of::<DocumentTermInfo>()*self.docs.len() as usize)};
        Cow::Borrowed(vals)
    }

    fn get_mem_buffer_type() -> i32 {
        10
    }
}


pub struct RAMFilesystem {
    entries: std::collections::HashMap<String,Vec<u8>>,
    mem_size: usize,
}

impl RAMFilesystem {
    pub fn new() -> RAMFilesystem {
        RAMFilesystem {
            entries: std::collections::HashMap::new(),
            mem_size: 0
        }
    }

    pub fn memory_length(&self) -> usize {
        self.mem_size
    }

    pub fn persist(&self, base_path: &str) -> Result<(),std::io::Error> {
        for (part_path,data) in self.entries.iter() {
            let path = std::path::Path::new(&base_path).join(part_path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path,&data)?;
        }
        Ok(())
    }

    pub fn from_disk(path: &str) -> Result<RAMFilesystem,std::io::Error> {
        let mut mem_size = 0;
        let mut entries = std::collections::HashMap::new();
        for entry in WalkDir::new(path) {
            let paths = entry.unwrap();
            if !paths.file_type().is_dir() {
                let result = std::fs::read(paths.path())?;
                mem_size+=result.len();
                entries.insert(paths.path().to_str().unwrap()[path.len()+1..].to_string(), result);
            }
        }
        Ok(RAMFilesystem {
            entries,
            mem_size
        })
    }
}


impl Filesystem for RAMFilesystem {
    fn write_file(&mut self, path: &str, slice: &[u8]) -> Result<(),std::io::Error> {
        if let Some(vec) = self.entries.get_mut(path) {
            self.mem_size-=vec.len();
            vec.clear();
            vec.extend_from_slice(slice);
            self.mem_size+=slice.len();
        }
        else {
            self.entries.insert(path.to_string(), slice.to_vec());
            self.mem_size+=slice.len();
        }
        Ok(())
    }

    fn load_file(&self,path: &str) -> Result<&[u8],std::io::Error> {
        if let Some(vec) = self.entries.get(path) {
            return Ok(&vec);
        }
        Err(std::io::Error::new(std::io::ErrorKind::NotFound,"Could not find file!"))
    }
}

struct BufferedVector<'a> {
    data: &'a mut Vec<u8>,
}

impl<'a> BufferedVector<'a> {
    fn new(target: &'a mut Vec<u8>) -> BufferedVector {
        BufferedVector {
            data: target
        }
    }
}

impl<'a> Write for BufferedVector<'a> {
    fn write(&mut self, buf: &[u8]) -> Result<usize,std::io::Error> {
        self.data.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(),std::io::Error> {
        Ok(())
    }
}

pub struct MMapedFilesystem {
    base_dir: String,
    mapped_files: std::collections::HashMap<String, Box<dyn std::ops::Deref<Target=[u8]>>>,
}

impl MMapedFilesystem {
    pub fn from(base_dir: &str) -> Result<MMapedFilesystem,std::io::Error> {
        let mut entries: std::collections::HashMap<String, Box<dyn std::ops::Deref<Target=[u8]>>> = std::collections::HashMap::new();
        for entry in WalkDir::new(base_dir) {
            let paths = entry.unwrap();
            if !paths.file_type().is_dir() {
                let result = unsafe{Mmap::map(&File::open(paths.path())?)?};
                entries.insert(paths.path().to_str().unwrap().to_string(), Box::new(result));
            }
        }
 
        Ok(MMapedFilesystem {
            base_dir: base_dir.to_string(),
            mapped_files: entries
        })
    }
}

impl Filesystem for MMapedFilesystem {
    fn write_file(&mut self, path_orig: &str, slice: &[u8]) -> Result<(),std::io::Error> {
        let path = std::path::Path::new(&self.base_dir).join(path_orig);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path.clone(),slice)?;
        if slice.len() == 0 {
            self.mapped_files.insert(path_orig.to_string(),Box::new(Vec::new()));
        }
        else {
            let result = unsafe{Mmap::map(&File::open(path)?)?};
            self.mapped_files.insert(path_orig.to_string(), Box::new(result));
        }
        Ok(())
    }

    fn load_file(&self,path: &str) -> Result<&[u8],std::io::Error> {
        if let Some(vec) = self.mapped_files.get(path) {
            return Ok(&vec);
        }
        Err(std::io::Error::new(std::io::ErrorKind::NotFound,"Could not find file!"))
    }
}

pub struct Token {
    pub key: String,
    pub start: usize,
    pub end: usize,
}

pub trait TokenStreamer {
    fn next(&mut self) -> Option<&Token>;
}

pub trait Tokenizer<'a> {
    fn stream(&mut self, string: &str) -> Box<dyn TokenStreamer>;
    fn map(&mut self, map_func: Box<dyn FnMut(&mut Token) -> bool+'a>);
}

pub struct StandardTokenizer<'a> {
    maps: Vec<Box<dyn FnMut(&mut Token) -> bool+'a>>,
}


impl<'a> Tokenizer<'a> for StandardTokenizer<'a> {
    fn stream(&mut self, string: &str) -> Box<dyn TokenStreamer> {
        let regex = Regex::new(r"\b\S+\b").unwrap();
        let mut matches = regex.find_iter(string);
        let mut tokens = Vec::new();
        'outer: while let Some(item) = matches.next() {
            let mut token = Token {
                key: item.as_str().to_owned(),
                start: item.start(),
                end: item.end()
            };

            for x in self.maps.iter_mut() {
                if !x(&mut token) {
                    continue 'outer;
                }
            }
            tokens.push(token);
        }
        Box::new(StandardTokenStreamer {
            tokens: tokens,
            index: 0,
        })
    }
    
    fn map(&mut self, map_func: Box<dyn FnMut(&mut Token) -> bool+'a>) {
        self.maps.push(map_func);
    }
}

pub struct RawTokenizer<'a> {
    maps: Vec<Box<dyn FnMut(&mut Token) -> bool+'a>>,
}

impl<'a> RawTokenizer<'a> {
    pub fn new() -> RawTokenizer<'a> {
        RawTokenizer {
            maps: Vec::new()
        }
    }
}

impl<'a> Tokenizer<'a> for RawTokenizer<'a> {
    fn stream(&mut self, string: &str) -> Box<dyn TokenStreamer> {
        let mut token = Token {
            key: string.to_owned(),
            start: 0,
            end: string.len()
        };

        for x in self.maps.iter_mut() {
            if !x(&mut token) {
                return Box::new(StandardTokenStreamer {
                    tokens: Vec::new(),
                    index: 0,
                });
            }
        }
        Box::new(StandardTokenStreamer {
            tokens: vec![token],
            index: 0,
        })
    }
    
    fn map(&mut self, map_func: Box<dyn FnMut(&mut Token) -> bool+'a>) {
        self.maps.push(map_func);
    }
}

pub struct StandardTokenStreamer {
    tokens: Vec<Token>,
    index: usize,
}

impl TokenStreamer for StandardTokenStreamer {
    fn next(&mut self) -> Option<&Token> {
        let ret = self.tokens.get(self.index);
        self.index+=1;
        ret
    }
}

pub fn to_lowercase(token: &mut Token) -> bool {
    token.key = token.key.to_lowercase();
    true
}

pub fn filter_long(token: &mut Token) -> bool {
    token.key.len()<30
}




#[derive(Serialize,Deserialize)]
pub struct Index<'a> {
    term_index: std::collections::BTreeMap<String,std::collections::BTreeMap<u64,Vec<u32>>>,
    indexed_lengths: std::collections::BTreeMap<u64,u32>,
    total_indexed_count: u32,
    index_num: i32,
    #[serde(skip)]
    #[serde(default="Index::create_boxed_tokenizer")]
    tokenizer: Box<dyn Tokenizer<'a>+'a>,
}

impl<'a> Index<'a> {
    fn create_standard_tokenizer() -> StandardTokenizer<'a> {
        let mut tokenizer = StandardTokenizer {
            maps: Vec::new()
        };
        tokenizer.map(Box::new(to_lowercase));
        tokenizer.map(Box::new(filter_long));
        tokenizer
    }
    fn create_boxed_tokenizer() -> Box<StandardTokenizer<'a>> {
        Box::new(Index::create_standard_tokenizer())
    }

    pub fn new(index_num: i32) -> Index<'a> {
        Index {
            term_index: std::collections::BTreeMap::new(),
            indexed_lengths: std::collections::BTreeMap::new(),
            total_indexed_count: 0,
            index_num,
            tokenizer: Box::new(Index::create_standard_tokenizer())
        }
    }

    pub fn set_tokenizer<T: Tokenizer<'a>+'a>(&mut self, tokenizer: T) {
        self.tokenizer = Box::new(tokenizer);
    }

    fn get_index_bincode_path(index_num: i32) -> String {
        index_num.to_string()+"_index.bincode"
    }

    fn get_index_automaton_path(index_num: i32) -> String {
        index_num.to_string()+"_fst.bin"
    }

    fn get_index_term_path(index_num: i32) -> String {
        index_num.to_string()+"_term_index.bin"
    }


    pub fn from_fs<F: Filesystem>(index_num: i32, fs: &mut F) -> Result<Index<'a>,std::io::Error> {
        let result = fs.load_file(&Index::get_index_bincode_path(index_num))?;
        Ok(bincode::deserialize(&result).unwrap())
    }

    pub fn to_fs<F: Filesystem>(&self, fs: &mut F) -> Result<(),std::io::Error> {
        fs.write_file(&Index::get_index_bincode_path(self.index_num), &bincode::serialize(self).unwrap())
    }

    fn add_key(&mut self, key:  &str, doc_id: u64, index: u32) {
        if let Some(idx) = self.term_index.get_mut(key) {
            if let Some(entry) = idx.get_mut(&doc_id) {
                entry.push(index as u32);
            }
            else {
                idx.insert(doc_id,vec![index as u32]);
            }
        }
        else {
            let mut pos_list = std::collections::BTreeMap::new();
            pos_list.insert(doc_id,vec![index as u32]);
            self.term_index.insert(key.to_string(),pos_list);
        }
    }

    pub fn index_field<'b>(&'b mut self, body: &'b str, doc_id: u64) {
        let mut stream = self.tokenizer.stream(body);
        
        let mut count = 0;
        while let Some(token) = stream.next() {
            count+=1;
            self.add_key(&token.key, doc_id,token.start as u32);
        }
        self.indexed_lengths.insert(doc_id, count);
        self.total_indexed_count+=1;
    }

    pub fn commit<F: Filesystem>(&mut self, fs: &mut F) -> Result<(),std::io::Error> {
        // This is where we'll write our map to.
        let mut target = Vec::new();
        let wtr  = BufferedVector::new(&mut target);

        // Create a builder that can be used to insert new key-value pairs.
        let mut build = MapBuilder::new(wtr).unwrap();
        let mut index_data: Vec<u8> = Vec::new();

        for (key,value) in self.term_index.iter() {
            build.insert(&key, index_data.len() as u64).unwrap();
            let mut documents : Vec<DocumentTermInfo>  = Vec::new();

            let mut writer2 = MemBufferWriter::new();
            for (x,positions) in value {
                documents.push(DocumentTermInfo {
                    doc_ptr: *x,
                    doc_freq: (positions.len() as f32/(*self.indexed_lengths.get(x).unwrap() as f32))*(self.total_indexed_count as f32 / positions.len() as f32),
                });
                writer2.add_entry(&positions[..]);
            }

            let writer = DocumentTermCollection {
                docs: &documents
            };

            writer2.add_entry(writer);
            index_data.extend_from_slice(&writer2.finalize());
        }


        // Finish construction of the map and flush its contents to disk.
        build.finish().unwrap();
        
        fs.write_file(&Index::get_index_term_path(self.index_num),&index_data)?;
        fs.write_file(&Index::get_index_automaton_path(self.index_num),&target)?;
        self.to_fs(fs)?;
        Ok(())
    }
}

#[derive(Serialize,Deserialize)]
struct IndexWriterInfo {
    num_indexes: usize
}

pub struct IndexWriter<'a,T: Filesystem> {
    fs: &'a mut T,
    document_data: Vec<u8>,
    indexes: Vec<Index<'a>>,
    index_locked: bool,
}


impl<'a,T: Filesystem> IndexWriter<'a,T> {
    pub fn add_index<X: Into<i32>>(&mut self, doc_index: X) -> &mut Index<'a> {
        if self.index_locked {
            panic!("The index is already locked a documents have been added! Please recreate the index if you want to add another fields for indexing");
        }

        let next_index = doc_index.into();
        if next_index != self.indexes.len() as i32 {
            panic!("The index values must be supplied continous and starting from zero! When using an enum this means using the fields in the order they are in the enum!");
        }

        self.indexes.push(Index::new(next_index));
        self.indexes.last_mut().unwrap()
    }

    fn get_document_path() -> &'a str {
        "documents.bin"
    }

    fn get_index_writer_info_path() -> &'a str {
        "index_info.bincode"
    }

    pub fn from_fs(filesystem: &mut T) -> Result<IndexWriter<T>,std::io::Error> {
        let (document_data,index_locked) = if let Ok(val) = filesystem.load_file(IndexWriter::<MMapedFilesystem>::get_document_path()) {
            (val.to_vec(),true)
        }
        else {
            (Vec::new(),false)
        };

        let indicies = if document_data.len() > 0 {
            let num_indicies : IndexWriterInfo = bincode::deserialize(filesystem.load_file(IndexWriter::<MMapedFilesystem>::get_index_writer_info_path())?).unwrap();
            let mut new_indicies = Vec::with_capacity(num_indicies.num_indexes);
            for i in 0..num_indicies.num_indexes {
                new_indicies.push(Index::from_fs(i as i32, filesystem)?);
            }
            new_indicies
        }
        else {
            Vec::new()
        };

        Ok(IndexWriter {
            fs: filesystem,
            document_data: document_data,
            indexes: indicies,
            index_locked: index_locked,
        })
    }

    pub fn add_document<'b,X: Serialize+Deserialize<'b>>(&mut self, x: Document<'b,X>) {
        self.index_locked = true;
        let doc_id = self.document_data.len() as u64;
        let mut body = MemBufferWriter::new();
        
        for (index_target,_) in x.indexable_parts.iter() {
            if *index_target >= self.indexes.len() as i32 {
                panic!("Wrong index in Document the maximum index for Documents is {}",self.indexes.len());
            }
        }
        for target_index in 0..self.indexes.len() {
            if let Some(value) = x.indexable_parts.get(&(target_index as i32)) {
                body.add_entry(*value);
                self.indexes[target_index].index_field(*value,doc_id);
            }
            else {
                body.add_entry("");
            }
        }
        body.add_serde_entry(&x.metadata);
        self.document_data.extend_from_slice(&body.finalize());
    }

    pub fn commit(&mut self) -> Result<(),std::io::Error> {
        for index in self.indexes.iter_mut() {
            index.commit(self.fs)?;
        }
        self.fs.write_file(IndexWriter::<MMapedFilesystem>::get_document_path(), &self.document_data)?;

        let index_info = IndexWriterInfo {
            num_indexes: self.indexes.len()
        };
        self.fs.write_file(IndexWriter::<MMapedFilesystem>::get_index_writer_info_path(), &bincode::serialize(&index_info).unwrap())?;
        Ok(())
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


pub enum QueryOperation {
    Or,
    And
}

enum QueryOperationSettings {
    LevenstheinDistance1,
    LevenstheinDistance2,
    StartsWith,
    Exact,
}


pub struct Query<'a> {
    query: &'a str,
    query_settings: QueryOperationSettings,
    boost_factor: f32,
    target_index: Vec<i32>,
}
        
lazy_static! { 
static ref AUTOMATON_DIST_1: levenshtein_automata::LevenshteinAutomatonBuilder = LevenshteinAutomatonBuilder::new(1, true);  
static ref AUTOMATON_DIST_2: levenshtein_automata::LevenshteinAutomatonBuilder = LevenshteinAutomatonBuilder::new(2, true);
}

impl<'a> Query<'a> {
    pub fn exact(query: &'a str) -> Query<'a> {
        Query::create_query_with_settings(query, QueryOperationSettings::Exact)
    }

    fn create_query_with_settings(query: &'a str, settings: QueryOperationSettings) -> Query<'a> {
        Query {
            query: query,
            query_settings: settings,
            target_index: Vec::new(),
            boost_factor: 1.0,
        }
    }

    pub fn target<T: Into<i32>>(mut self,target: T) -> Query<'a> {
        self.target_index.push(target.into());
        self
    }

    pub fn starts_with(query: &'a str) -> Query<'a> {
        Query::create_query_with_settings(query, QueryOperationSettings::StartsWith)
    }

    pub fn boost(mut self, factor: f32) -> Query<'a> {
        self.boost_factor = factor;
        self
    }

    pub fn fuzzy(query: &'a str, max_distance: u8) -> Query {
        if max_distance == 1 {
            Query::create_query_with_settings(query, QueryOperationSettings::LevenstheinDistance1)
        }
        else {
            Query::create_query_with_settings(query, QueryOperationSettings::LevenstheinDistance2)
        }
    }
}


struct OrderedCollector<'a,X: Serialize+Deserialize<'a>> {
    traversal: Vec<SearchHit<'a,X>>,
    load_metadata: bool,
    index: i32,
    boost_factor: f32,
}

impl<'a,X: Serialize+Deserialize<'a>> OrderedCollector<'a,X> {
    pub fn new(load_metadata: bool, index: i32, boost: f32) -> OrderedCollector<'a,X> {
        OrderedCollector {
            traversal: Vec::new(),
            load_metadata: load_metadata,
            index: index,
            boost_factor: boost
        }
    }

    pub fn finalize(self) -> Vec<SearchHit<'a,X>> {
        self.traversal
    }

    pub fn create_one<'b: 'a>(&self,idx: &'a DocumentTermInfo, word_index: std::rc::Rc<(String,u64,u8)>, word_pos: &'a [u32], buffer: &'a [u8]) -> SearchHit<'a,X> {
        let information = MemBufferReader::new(&buffer[idx.doc_ptr as usize..]).unwrap();
        let meta = if self.load_metadata {
            Some(information.load_serde_entry::<X>(information.len()-1).unwrap())
        }
        else {
            None
        };

        let mut search = SearchHit {
            doc_ptr: idx.doc_ptr,
            metadata: meta,
            doc_score: self.boost_factor*idx.doc_freq/((word_index.2+1) as f32),
            index_hits: Vec::with_capacity(information.len()-1)
        };

        for y in 0..(information.len()-1) {
            if y as i32 == self.index {
                search.index_hits.push(IndexHit {
                    content: information.load_entry(y).unwrap(),
                    matched_words: vec![word_index.clone()],
                    positions: vec![word_pos],
                });
            }
            else {
                search.index_hits.push(IndexHit {
                    content: information.load_entry(y).unwrap(),
                    matched_words: Vec::new(),
                    positions: Vec::new(),
                });
            }
        }
        search
    }

    pub fn insert_all<'b: 'a>(&mut self,idx: &'a [DocumentTermInfo], word_index: std::rc::Rc<(String,u64,u8)>, word_pos: &[&'a [u32]], buffer: &'a [u8]) {
        for x in 0..idx.len() {
            self.traversal.push(self.create_one(&idx[x], word_index.clone(), word_pos[x], buffer));
        }
    }
    
    pub fn add_array<'b: 'a>(&mut self, idx: &'a [DocumentTermInfo], word_index: std::rc::Rc<(String,u64,u8)>, word_pos: Vec<&'a [u32]>, buffer: &'a [u8]) {
        let mut index_own = 0;
        let mut index_other = 0;
        loop {
            if index_own == self.traversal.len() {
                self.insert_all(&idx[index_other..], word_index.clone(), &word_pos[index_other..], buffer);
                return ();
            }
            
            if index_other == idx.len() {
                return ();
            }

            let curr_ptr = self.traversal[index_own].doc_ptr;
            if curr_ptr < idx[index_other].doc_ptr {
                index_own+=1;
                continue;
            }
            else if curr_ptr == idx[index_other].doc_ptr {
                let ptr = &mut self.traversal[index_own].index_hits[self.index as usize];
                ptr.matched_words.push(word_index.clone());
                ptr.positions.push(word_pos[index_other]);
                self.traversal[index_own].doc_score+=self.boost_factor*idx[index_other].doc_freq/((word_index.2+1) as f32);
                index_own+=1;
            }
            else {
                let hit = self.create_one(&idx[index_other], word_index.clone(), word_pos[index_other], buffer);
                self.traversal.insert(index_own, hit);
            }

            index_other+=1;
        }
    }
}



pub struct IndexHit<'a> {
    pub content: &'a str,
    pub matched_words: Vec<std::rc::Rc<(String,u64,u8)>>,
    pub positions: Vec<&'a [u32]>,
}


pub struct SearchHit<'a,X: Serialize+Deserialize<'a>> {
    pub doc_ptr: u64,
    pub metadata: Option<X>,
    pub doc_score: f32,
    pub index_hits: Vec<IndexHit<'a>>,
}

pub struct DocumentSearchResult<'a,X: Serialize+Deserialize<'a>> {
    pub hits: Vec<SearchHit<'a,X>>,
}

impl<'a,X:Serialize+Deserialize<'a>> DocumentSearchResult<'a,X> {
    pub fn sort_by_score(&mut self) {
        self.hits.sort_by(|x,y|{x.doc_score.partial_cmp(&y.doc_score).unwrap_or(std::cmp::Ordering::Equal)});
    }

    pub fn limit(&mut self, limit: usize) {
        if self.hits.len() > limit {
            self.hits.drain(limit..);
        }
    }

    pub fn and(mut self, mut other: DocumentSearchResult<'a,X>) -> DocumentSearchResult<'a,X> {
        let mut index_own = 0;
        let mut index_other = 0;

        loop {
            if index_own == self.hits.len() {
                return self;
            }
            if index_other>=other.hits.len() {
                self.hits.drain(index_own..);
                return self;
            }

            if self.hits[index_own].doc_ptr < other.hits[index_other].doc_ptr {
                self.hits.remove(index_own);
                continue;
            }
            else if self.hits[index_own].doc_ptr == other.hits[index_other].doc_ptr {
                self.hits[index_own].doc_score += other.hits[index_other].doc_score;
                for hit in 0..other.hits[index_other].index_hits.len() {
                    self.hits[index_own].index_hits[hit].positions.append(&mut other.hits[index_other].index_hits[hit].positions);
                    self.hits[index_own].index_hits[hit].matched_words.append(&mut other.hits[index_other].index_hits[hit].matched_words);
                }
                index_own+=1;
            }
            index_other+=1;
        }
    }

    pub fn or(mut self, mut other: DocumentSearchResult<'a,X>) -> DocumentSearchResult<'a,X> {
        let mut index_own = 0;
        let mut index_other = 0;

        loop {
            if index_own == self.hits.len() {
                self.hits.extend(other.hits.drain(index_other..));
                return self;
            }
            if index_other>=other.hits.len() {
                return self;
            }

            if self.hits[index_own].doc_ptr < other.hits[index_other].doc_ptr {
                index_own+=1;
                continue;
            }
            else if self.hits[index_own].doc_ptr == other.hits[index_other].doc_ptr {
                self.hits[index_own].doc_score += other.hits[index_other].doc_score;
                for hit in 0..other.hits[index_other].index_hits.len() {
                    self.hits[index_own].index_hits[hit].positions.append(&mut other.hits[index_other].index_hits[hit].positions);
                    self.hits[index_own].index_hits[hit].matched_words.append(&mut other.hits[index_other].index_hits[hit].matched_words);
                }
                index_own+=1;
            }
            else {
                let new_hit = SearchHit {
                    doc_ptr: other.hits[index_other].doc_ptr,
                    doc_score: other.hits[index_other].doc_score,
                    metadata: other.hits[index_other].metadata.take(),
                    index_hits: other.hits[index_other].index_hits.drain(..).collect(),
                };
                self.hits.insert(index_own, new_hit);
            }
            index_other+=1;
        }
    }
}

pub struct IndexReader<'a> {
    data: &'a [u8],
    index_data: std::collections::HashMap<i32,(&'a [u8], &'a [u8])>,
    load_metadata: bool,
}


impl<'a> IndexReader<'a> {
    pub fn from_fs<F: Filesystem, T: Into<i32>>(filesystem: &F, indexes: Vec<T>) -> Result<IndexReader,std::io::Error> {
        let data = filesystem.load_file(IndexWriter::<MMapedFilesystem>::get_document_path())?;
        let mut index_data = std::collections::HashMap::new();
        for x in indexes.into_iter() {
            let index_i32: i32 = x.into();
            index_data.insert(index_i32, (filesystem.load_file(&Index::get_index_automaton_path(index_i32))?,filesystem.load_file(&Index::get_index_term_path(index_i32))?));
        }

        Ok(IndexReader {
            data: data,
            index_data: index_data,
            load_metadata: false,
        })
    }

    pub fn load_metadata(&mut self, load_metadata: bool) {
        self.load_metadata = load_metadata;
    }

    pub fn search<X: Serialize+Deserialize<'a>>(&'a self, query: Query<'a>) -> Result<DocumentSearchResult<'a,X>,std::io::Error> {
        self.search_query(query)
    }

    pub fn search_query<X: Serialize+Deserialize<'a>>(&'a self, mut query: Query<'a>) -> Result<DocumentSearchResult<'a,X>,std::io::Error> {
        let mut result: DocumentSearchResult<'a,X> = DocumentSearchResult {
            hits: Vec::new()
        };
        
        if query.target_index.len() == 0 {
            query.target_index = self.index_data.keys().map(|x|*x).collect();
        }

        for index in query.target_index.iter() {
            if let Some((index_automaton,index_data)) = self.index_data.get(&index) {
                result = result.or(match query.query_settings {
                    QueryOperationSettings::Exact => {
                        self.load_hits(
                            self.do_search_in_index(fst::automaton::Str::new(query.query),index_automaton)?,
                            *index,
                            index_data,
                            query.boost_factor
                        )
                    },
                    QueryOperationSettings::LevenstheinDistance1 => {
                        self.load_hits(
                            self.do_search_in_index_levensthein(query.query,1,index_automaton)?,
                            *index,
                            index_data,
                            query.boost_factor
                        )
                    },
                    QueryOperationSettings::LevenstheinDistance2 => {
                        self.load_hits(
                            self.do_search_in_index_levensthein(query.query,2,index_automaton)?,
                            *index,
                            index_data,
                            query.boost_factor
                        )
                    },
                    QueryOperationSettings::StartsWith => {
                        self.load_hits(
                            self.do_search_in_index(fst::automaton::Str::new(query.query).starts_with(),index_automaton)?,
                            *index,
                            index_data,
                            query.boost_factor
                        )
                    }
                }?);
            }
            else {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound,format!("Could not find index or index is not loaded \"{}\"",index)));
            }
        }
        Ok(result)
    }

    fn do_search_in_index_levensthein(&'a self,  query: &'a str, distance: u8, index_automaton: &'a [u8]) -> Result<Vec<std::rc::Rc<(String,u64,u8)>>,std::io::Error> {
        let load_automaton = fst::Map::new(index_automaton).unwrap();
        let auto = if distance == 1 {
            DFAWrapper(AUTOMATON_DIST_1.build_dfa(query))
        }
        else {
            DFAWrapper(AUTOMATON_DIST_2.build_dfa(query))
        };

        let mut result = load_automaton.search_with_state(&auto).into_stream();
        
        let mut sorted_states:Vec<std::rc::Rc<(String,u64,u8)>> = Vec::with_capacity(100);
        while let Some((key_u8,value,state)) = result.next() {
            let key = unsafe{std::str::from_utf8_unchecked(key_u8).to_string()};
            match auto.0.distance(state) {
                Distance::Exact(a) => {sorted_states.push(std::rc::Rc::new((key,value,a)))},
                _ => {}
            }
        }
        sorted_states.sort_by(|x,y| {x.2.cmp(&y.2)});
        Ok(sorted_states)
    }

    fn do_search_in_index<A: fst::Automaton>(&'a self, automaton: A,index_automaton: &'a [u8]) -> Result<Vec<std::rc::Rc<(String,u64,u8)>>,std::io::Error> {
        let load_automaton = fst::Map::new(index_automaton).unwrap();

        let mut return_value = Vec::with_capacity(100);
        let mut result = load_automaton.search(automaton).into_stream();
        while let Some((key_u8,value)) = result.next() {
            let key = unsafe{std::str::from_utf8_unchecked(key_u8).to_string()};
            return_value.push(std::rc::Rc::new((key,value,0)));
        }
        Ok(return_value)
    }

    pub fn load_hits<X: Serialize+Deserialize<'a>>(&'a self, matches: Vec<std::rc::Rc<(String,u64,u8)>>, index: i32, val: &'a [u8], boost_factor: f32) -> Result<DocumentSearchResult<'a, X>,std::io::Error> 
        {
        let mut orderedcoll: OrderedCollector<'a,X> = OrderedCollector::new(self.load_metadata, index, boost_factor);

        let mut counter = 0;
        for value in matches.iter() {
            let reader = MemBufferReader::new(&val[value.1 as usize..]).unwrap();
            let val: DocumentTermCollection = reader.load_entry(reader.len()-1).unwrap();
            let mut docs : Vec<&'a [u32]> = Vec::with_capacity(val.docs.len());

            for x in 0..val.docs.len() {
                docs.push(reader.load_entry(x).unwrap());
            }

            orderedcoll.add_array(val.docs, matches[counter].clone(), docs, self.data);
            counter+=1;
        }
        Ok(DocumentSearchResult{ hits: orderedcoll.finalize()})
    }
}

#[cfg(test)]
mod tests {
    use super::{Document,RAMFilesystem,Filesystem,Index,IndexWriter,Query,MMapedFilesystem,IndexReader,RawTokenizer,Tokenizer,filter_long};
    use serde::{Serialize,Deserialize};
    
    #[derive(Serialize,Deserialize)]
    struct DocumentMeta<'a> {
        title: &'a str,
        path: &'a str,
    }

    #[test]
    fn check_document_integrity() {
        let new_meta = DocumentMeta {
            title: "Intel developers system manual",
            path: "main.txt"
        };
        let mut doc = Document::new(&new_meta);
        assert_eq!(bincode::serialize(&doc.metadata).unwrap(),bincode::serialize(&new_meta).unwrap());

        doc.add_field(0, "The fox walked right up to the base");
        assert_eq!(doc.indexable_parts.len(),1);
        assert_eq!(doc.indexable_parts.get(&0).unwrap(),&"The fox walked right up to the base");

        doc.add_field(1, "main body");
        assert_eq!(doc.indexable_parts.len(),2);
        assert_eq!(doc.indexable_parts.get(&0).unwrap(),&"The fox walked right up to the base");
        assert_eq!(doc.indexable_parts.get(&1).unwrap(),&"main body");
    }

    #[should_panic]
    #[test]
    fn check_document_multi_fields() {
        let new_meta = DocumentMeta {
            title: "Intel developers system manual",
            path: "main.txt"
        };
        let mut doc = Document::new(&new_meta);
        assert_eq!(bincode::serialize(&doc.metadata).unwrap(),bincode::serialize(&new_meta).unwrap());

        doc.add_field(0, "The fox walked right up to the base");
        assert_eq!(doc.indexable_parts.len(),1);
        assert_eq!(doc.indexable_parts.get(&0).unwrap(),&"The fox walked right up to the base");

        doc.add_field(0, &"main body");
    }

    #[test]
    pub fn check_ramfs() {
        let mut ramdir = RAMFilesystem::new();
        let result = ramdir.write_file("fuchs", &[0,10,0,100,200]);
        assert_eq!(result.is_err(),false);
        assert_eq!(ramdir.memory_length(), 5);
        assert_eq!(ramdir.load_file("fuchs").unwrap(),[0,10,0,100,200]);
        assert_eq!(ramdir.load_file("fuchss").is_err(),true);
    }

    #[test]
    pub fn check_ramfs_double_write() {
        let mut ramdir = RAMFilesystem::new();
        let result = ramdir.write_file("fuchs", &[0,10,0,100,200]);
        assert_eq!(result.is_err(),false);
        assert_eq!(ramdir.memory_length(), 5);
        assert_eq!(ramdir.load_file("fuchs").unwrap(),&[0,10,0,100,200]);
        assert_eq!(ramdir.load_file("fuchss").is_err(),true);
        ramdir.write_file("fuchsd", &[0,10,0,100,200]).unwrap();
        ramdir.write_file("fuchs", &[0]).unwrap();
        assert_eq!(ramdir.memory_length(), 6);
        assert_eq!(ramdir.load_file("fuchs").unwrap(),&[0]);
        assert_eq!(ramdir.load_file("fuchsd").unwrap(),&[0,10,0,100,200]);
    }

    #[test]
    pub fn check_index() {
        let mut new_index = Index::new(0);
        new_index.index_field("der Hund sprach zum Fuchs", 0);
        assert_eq!(new_index.term_index.len(),5);
        assert_eq!(*new_index.term_index.get("hund").unwrap().get(&0).unwrap(),vec![4]);
        assert_eq!(new_index.term_index.get("fuchs").is_some(),true);

        new_index.index_field("Das Kaptial", 1);
        assert_eq!(new_index.term_index.len(),7);
        assert_eq!(*new_index.term_index.get("das").unwrap().get(&1).unwrap(),vec![0]);

        assert_eq!(new_index.total_indexed_count, 2);
        assert_eq!(*new_index.indexed_lengths.get(&0).unwrap(),5);
        assert_eq!(*new_index.indexed_lengths.get(&1).unwrap(),2);
    }

    #[should_panic]
    #[test]
    pub fn check_multi_index_not_continous() {
        let mut ram = RAMFilesystem::new();
        let mut index = IndexWriter::from_fs(&mut ram).unwrap();
        index.add_index(0);
        index.add_index(2);
    }

    #[should_panic]
    #[test]
    pub fn check_multi_index_not_starting_with_zero() {
        let mut ram = RAMFilesystem::new();
        let mut index = IndexWriter::from_fs(&mut ram).unwrap();
        index.add_index(1);
    }

    #[derive(Serialize,Deserialize)]
    struct NoMeta {}

    #[should_panic]
    #[test]
    pub fn check_multi_index_add_doc_with_wrong_index() {
        let no_meta = NoMeta{};
        let mut ram = RAMFilesystem::new();
        let mut index = IndexWriter::from_fs(&mut ram).unwrap();
        index.add_index(0);
        let mut doc = Document::new(&no_meta);
        doc.add_field(1, "New content");
        index.add_document(doc);
    }

    #[test]
    pub fn check_multi_index_add_doc() {
        let mut ram = RAMFilesystem::new();
        let mut index = IndexWriter::from_fs(&mut ram).unwrap();
        index.add_index(0);
        let no_meta = NoMeta{};
        let mut doc = Document::new(&no_meta);
        doc.add_field(0, "New content");
        index.add_document(doc);

        assert_eq!(index.indexes.len(),1);
        assert_eq!(index.index_locked,true);
        assert_ne!(index.document_data.len(),0);
    }

    enum IndexEnum {
        Body,
        Title,
    }

    impl Into<i32> for IndexEnum {
        fn into(self) -> i32 {
            self as i32
        }
    }



    #[test]
    pub fn check_writer_index() {
        let mut ram = RAMFilesystem::new();
        let mut index = IndexWriter::from_fs(&mut ram).unwrap();
        index.add_index(IndexEnum::Body);
        index.add_index(IndexEnum::Title);

        let new_meta = DocumentMeta {
            title: "Intel developers system manual",
            path: "main.txt"
        };

        let mut new_doc = Document::new(&new_meta);
        new_doc.add_field(IndexEnum::Title, "hello how are you?");
        index.add_document(new_doc);
        let val = index.commit();
        assert_eq!(val.is_err(),false);
        assert_eq!(index.index_locked,true);
        assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,1);
        assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,0);
        assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),0);
        assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),4);
    }

    #[test]
    pub fn check_load_writer() {
        let mut ram = RAMFilesystem::new();
        {
            let mut index = IndexWriter::from_fs(&mut ram).unwrap();
            index.add_index(IndexEnum::Body);
            index.add_index(IndexEnum::Title);

            let new_meta = DocumentMeta {
                title: "Intel developers system manual",
                path: "main.txt"
            };

            let mut new_doc = Document::new(&new_meta);
            new_doc.add_field(IndexEnum::Title, "hello how are you?");
            index.add_document(new_doc);
            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
            assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,1);
            assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,0);
            assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),0);
            assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),4);
        }
        {
            let index = IndexWriter::from_fs(&mut ram).unwrap();
            assert_eq!(index.index_locked,true);
            assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,1);
            assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,0);
            assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),0);
            assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),4);
        }
    }

    #[should_panic]
    #[test]
    pub fn check_index_lock() {
        let mut ram = RAMFilesystem::new();
        let mut index = IndexWriter::from_fs(&mut ram).unwrap();
        index.add_index(IndexEnum::Body);
        index.add_index(IndexEnum::Title);

        let new_meta = DocumentMeta {
            title: "Intel developers system manual",
            path: "main.txt"
        };

        let mut new_doc = Document::new(&new_meta);
        new_doc.add_field(IndexEnum::Title, "hello how are you?");
        index.add_document(new_doc);
        index.add_index(2);
    }

    #[test]
    pub fn check_writer_real() {
        let _ = std::fs::remove_dir_all("fuchs");
        std::fs::create_dir("fuchs").unwrap();
        let mut mmaped = MMapedFilesystem::from("fuchs").unwrap();
        let mut index = IndexWriter::from_fs(&mut mmaped).unwrap();
        index.add_index(IndexEnum::Body);
        index.add_index(IndexEnum::Title);

        let new_meta = DocumentMeta {
            title: "Intel developers system manual",
            path: "main.txt"
        };

        let mut new_doc = Document::new(&new_meta);
        new_doc.add_field(IndexEnum::Title, "hello how are you?");
        index.add_document(new_doc);
        let val = index.commit();
        val.unwrap();
        //assert_eq!(val.is_err(),false);
        assert_eq!(index.index_locked,true);
        assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,1);
        assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,0);
        assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),0);
        assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),4);

        let mut ram = RAMFilesystem::from_disk("fuchs").unwrap();
        let second_load = IndexWriter::from_fs(&mut ram).unwrap();
        assert_eq!(second_load.index_locked,true);
        assert_eq!(second_load.indexes[IndexEnum::Title as usize].total_indexed_count,1);
        assert_eq!(second_load.indexes[IndexEnum::Body as usize].total_indexed_count,0);
        assert_eq!(second_load.indexes[IndexEnum::Body as usize].term_index.len(),0);
        assert_eq!(second_load.indexes[IndexEnum::Title as usize].term_index.len(),4);
    }

    #[test]
    pub fn check_query_boost() {
        let mut mmaped = RAMFilesystem::new();
        {
            let mut index = IndexWriter::from_fs(&mut mmaped).unwrap();
            index.add_index(IndexEnum::Body).set_tokenizer(RawTokenizer::new());
            index.add_index(IndexEnum::Title);

            let new_meta = DocumentMeta {
                title: "Intel developers system manual",
                path: "main.txt"
            };

            let mut new_doc0 = Document::new(&new_meta);
            new_doc0.add_field(IndexEnum::Title, "hollo");
            index.add_document(new_doc0);


            let mut new_doc = Document::new(&new_meta);
            new_doc.add_field(IndexEnum::Title, "hello how are you? Or hallo seems equally good. hollo");
            new_doc.add_field(IndexEnum::Body, "/alex/nice/");
            index.add_document(new_doc);

            let mut new_doc2 = Document::new(&new_meta);
            new_doc2.add_field(IndexEnum::Title, "This is a sad title with hello in it!");
            new_doc2.add_field(IndexEnum::Body, "/nick/nice/");
            index.add_document(new_doc2);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
            assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,3);
            assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,2);
            assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),2);
            assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),18);
        }

        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title,IndexEnum::Body]).unwrap();
        
        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query2 = Query::exact("hello").target(IndexEnum::Title).boost(2.0);
            let mut result = reader.search::<DocumentMeta>(query).unwrap();
            let mut result2 = reader.search::<DocumentMeta>(query2).unwrap();
            result.sort_by_score();
            result2.sort_by_score();
            assert_eq!(result.hits[0].doc_score*2.0, result2.hits[0].doc_score);
        }
        
        {
            let query = Query::fuzzy("hello",1).target(IndexEnum::Title);
            let mut result = reader.search::<DocumentMeta>(query).unwrap();
            result.sort_by_score();
            assert_eq!(result.hits.len(),3);
        }

        {
            let query = Query::fuzzy("hello",2).target(IndexEnum::Title);
            let mut result = reader.search::<DocumentMeta>(query).unwrap();
            result.sort_by_score();
            assert_eq!(result.hits.len(),3);
        }
    }

    #[test]
    pub fn check_multi_query() {
        let mut mmaped = RAMFilesystem::new();
        {
            let mut index = IndexWriter::from_fs(&mut mmaped).unwrap();
            index.add_index(IndexEnum::Body).set_tokenizer(RawTokenizer::new());
            index.add_index(IndexEnum::Title);

            let new_meta = DocumentMeta {
                title: "Intel developers system manual",
                path: "main.txt"
            };

            let mut new_doc = Document::new(&new_meta);
            new_doc.add_field(IndexEnum::Title, "hello how are you? Or hallo seems equally good.");
            new_doc.add_field(IndexEnum::Body, "/alex/nice/");
            index.add_document(new_doc);

            let mut new_doc2 = Document::new(&new_meta);
            new_doc2.add_field(IndexEnum::Title, "This is a sad title with hello in it!");
            new_doc2.add_field(IndexEnum::Body, "/nick/nice/");
            index.add_document(new_doc2);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
            assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,2);
            assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,2);
            assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),2);
            assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),17);
        }

        let mut reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title,IndexEnum::Body]).unwrap();
        
        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::starts_with("/alex/").target(IndexEnum::Body);
            let result = reader.search::<DocumentMeta>(query).unwrap();
            let result2 = reader.search::<DocumentMeta>(query_path).unwrap();
            let union = result.and(result2);
            assert_eq!(union.hits.len(), 1);
        }

        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::starts_with("/alex/").target(IndexEnum::Body);
            let result = reader.search::<DocumentMeta>(query).unwrap();
            let result2 = reader.search::<DocumentMeta>(query_path).unwrap();
            let or = result.or(result2);
            assert_eq!(or.hits.len(),2);
        }

        {
            let query = Query::fuzzy("hello",1).target(IndexEnum::Title);
            let query_path = Query::fuzzy("/alex/",1).target(IndexEnum::Body);
            let result = reader.search::<DocumentMeta>(query).unwrap();
            let result2 = reader.search::<DocumentMeta>(query_path).unwrap();
            let or = result.or(result2);
            assert_eq!(or.hits.len(),2);
        }

        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::exact("sad").target(IndexEnum::Title);
            let result = reader.search::<DocumentMeta>(query).unwrap();
            let result2 = reader.search::<DocumentMeta>(query_path).unwrap();
            let or = result2.or(result);
            assert_eq!(or.hits.len(),2);
        }

        {
            reader.load_metadata(false);
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::exact("sad").target(IndexEnum::Title);
            let result = reader.search::<DocumentMeta>(query).unwrap();
            let result2 = reader.search::<DocumentMeta>(query_path).unwrap();
            let or = result.and(result2);
            for x in or.hits.iter() {
                assert_eq!(x.metadata.is_none(),true);
            }
        }
    }

    #[test]
    pub fn check_raw_tokenizer() {
        let mut token = RawTokenizer::new();
        token.map(Box::new(filter_long));
        let mut stream = token.stream("asdsadasfgdsdsdsdskdskflkdsfdskkfdsklfllkdsflkdsflkdslkfdslk");
        assert_eq!(stream.next().is_none(),true);
        let mut second = token.stream("hallo welt");
        assert_eq!(second.next().unwrap().key,"hallo welt");
    }


    #[test]
    pub fn check_query() {
        let mut mmaped = RAMFilesystem::new();
        {
            let mut index = IndexWriter::from_fs(&mut mmaped).unwrap();
            index.add_index(IndexEnum::Body);
            index.add_index(IndexEnum::Title);

            let new_meta = DocumentMeta {
                title: "Intel developers system manual",
                path: "main.txt"
            };

            let mut new_doc = Document::new(&new_meta);
            new_doc.add_field(IndexEnum::Title, "hello how are you?");
            index.add_document(new_doc);

            let mut new_doc2 = Document::new(&new_meta);
            new_doc2.add_field(IndexEnum::Title, "This is a sad title with hello in it!");
            index.add_document(new_doc2);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
            assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,2);
            assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,0);
            assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),0);
            assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),12);
        }

        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title]).unwrap();
        let query = Query::exact("hello").target(IndexEnum::Title);
        let mut result = reader.search::<DocumentMeta>(query).unwrap();
        assert_eq!(result.hits.len(), 2);
        result.limit(1);
        assert_eq!(result.hits.len(), 1);

        
        let query2 = Query::exact("hello").target(IndexEnum::Body);
        let result2 = reader.search::<DocumentMeta>(query2);
        assert_eq!(result2.is_err(),true);
    }
}


#[cfg(feature="bench")]
mod bench {
    use test::Bencher;
    use super::{Document,IndexWriter,Query,MMapedFilesystem,IndexReader};
    use serde::{Serialize,Deserialize};

    #[derive(Serialize,Deserialize)]
    struct DocumentMeta<'a> {
        title: &'a str,
        path: &'a str,
    }

    enum IndexEnum {
        Body,
    }
    
    impl Into<i32> for IndexEnum {
        fn into(self) -> i32 {
            self as i32
        }
    }

    #[bench]
    fn check_reading_own(b: &mut Bencher) {
        let _ = std::fs::remove_dir_all("fuchs");
        std::fs::create_dir("fuchs").unwrap();
        let mut mmaped = MMapedFilesystem::from("fuchs").unwrap();

        {
            let mut new_writer = IndexWriter::from_fs(&mut mmaped).unwrap();
            new_writer.add_index(IndexEnum::Body);

            let hugestring = std::fs::read_to_string("main.txt").unwrap();

            let doc = DocumentMeta {
                title: "The modern Intel System Environment",
                path: "main.txt"
            };

            for _ in 0..10 {
                let mut document = Document::new(&doc);
                document.add_field(IndexEnum::Body, &hugestring);
                new_writer.add_document(document);
            }

            for _ in 0..100_000 {
                let mut document = Document::new(&doc);
                document.add_field(IndexEnum::Body,"and is indeed very nice write a small text about the live of brian to check how the time changes if i use a lot more text which should lead to more cache misses in theory");
                new_writer.add_document(document);
            }

            new_writer.commit().unwrap();
        }

        b.iter(|| {
            let mut reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Body]).unwrap();
            reader.load_metadata(false);
            let result = reader.search::<DocumentMeta>(Query::fuzzy("and",1).target(IndexEnum::Body)).unwrap();
            assert_eq!(result.hits.len(),100_010);
            let mut _score = 0.0;
            for x in result.hits.iter() {
                _score+=x.doc_score;
            }
        });
    }
}
