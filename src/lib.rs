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
pub use bincode;
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
                entries.insert(paths.path().strip_prefix(base_dir).unwrap().to_str().unwrap().to_string(), Box::new(result));
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
    fn next(&mut self) -> Option<Token>;
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
                    tokens: Vec::new()
                });
            }
        }
        Box::new(StandardTokenStreamer {
            tokens: vec![token]
        })
    }
    
    fn map(&mut self, map_func: Box<dyn FnMut(&mut Token) -> bool+'a>) {
        self.maps.push(map_func);
    }
}

pub struct StandardTokenStreamer {
    tokens: Vec<Token>,
}

impl TokenStreamer for StandardTokenStreamer {
    fn next(&mut self) -> Option<Token> {
        self.tokens.pop()
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

    fn add_key(&mut self, key:  String, doc_id: u64, index: u32) {
        if let Some(idx) = self.term_index.get_mut(&key) {
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
            self.term_index.insert(key,pos_list);
        }
    }

    pub fn index_field<'b>(&'b mut self, body: &'b str, doc_id: u64) {
        let mut stream = self.tokenizer.stream(body);
        
        let mut count = 0;
        while let Some(token) = stream.next() {
            count+=1;
            self.add_key(token.key, doc_id,token.start as u32);
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
            panic!("The index is already locked documents have been added! Please recreate the index if you want to add another fields for indexing");
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

    pub fn add_document<'b,X: Serialize+Deserialize<'b>>(&mut self, x: Document<'b,X>) -> u64 {
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
        doc_id
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


enum QueryOperationSettings {
    LevenstheinDistance1,
    LevenstheinDistance2,
    StartsWith,
    Exact,
    Subsequence
}


pub struct Query<'a> {
    query: &'a str,
    query_settings: QueryOperationSettings,
    query_id: usize,
    boost_factor: f32,
    target_index: Vec<(f32,i32)>,
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
            query_id: 0,
            boost_factor: 1.0,
        }
    }

    pub fn target<T: Into<i32>>(mut self,target: T) -> Query<'a> {
        self.target_index.push((1.0,target.into()));
        self
    }

    pub fn weighted_target<T: Into<i32>>(mut self,target: T, weight: f32) -> Query<'a> {
        self.target_index.push((weight,target.into()));
        self
    }

    pub fn starts_with(query: &'a str) -> Query<'a> {
        Query::create_query_with_settings(query, QueryOperationSettings::StartsWith)
    }

    pub fn subsequence(query: &'a str) -> Query<'a> {
        Query::create_query_with_settings(query, QueryOperationSettings::Subsequence)
    }

    pub fn boost(mut self, factor: f32) -> Query<'a> {
        self.boost_factor = factor;
        self
    }

    pub fn fuzzy(query: &'a str, max_distance: u8) -> Query {
        match max_distance {
            1 => Query::create_query_with_settings(query, QueryOperationSettings::LevenstheinDistance1),
            2 => Query::create_query_with_settings(query, QueryOperationSettings::LevenstheinDistance2),
            _ => panic!("Levensthein distance query is only implemented for distance 1 and 2 since the construction of the automaton takes exponential time"),
        }
    }
}


struct OrderedCollector<'a> {
    traversal: Vec<SearchHit<'a>>,
    index: i32,
    boost_factor: f32,
}

impl<'a> OrderedCollector<'a> {
    pub fn new(index: i32, boost: f32) -> OrderedCollector<'a> {
        OrderedCollector {
            traversal: Vec::new(),
            index: index,
            boost_factor: boost
        }
    }

    pub fn finalize(self) -> Vec<SearchHit<'a>> {
        self.traversal
    }

    pub fn create_one<'b: 'a>(&self,idx: &'a DocumentTermInfo, word_index: std::rc::Rc<(String,u64,u8,usize)>, word_pos: &'a [u32], buffer: &'a [u8]) -> SearchHit<'a> {
        let information = MemBufferReader::new(&buffer[idx.doc_ptr as usize..]).unwrap();
        let meta = information.load_entry::<&[u8]>(information.len()-1).unwrap();

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
                    hit_descriptions: vec![HitDescription{
                        word_info: word_index.clone(),
                        word_positions: word_pos
                    }]
                });
            }
            else {
                search.index_hits.push(IndexHit {
                    content: information.load_entry(y).unwrap(),
                    hit_descriptions: Vec::new(),
                });            
            }
        }
        search
    }

    pub fn insert_all<'b: 'a>(&mut self,idx: &'a [DocumentTermInfo], word_index: std::rc::Rc<(String,u64,u8,usize)>, word_pos: &[&'a [u32]], buffer: &'a [u8]) {
        for x in 0..idx.len() {
            self.traversal.push(self.create_one(&idx[x], word_index.clone(), word_pos[x], buffer));
        }
    }
    
    pub fn add_array<'b: 'a>(&mut self, idx: &'a [DocumentTermInfo], word_index: std::rc::Rc<(String,u64,u8,usize)>, word_pos: Vec<&'a [u32]>, buffer: &'a [u8]) {
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
                ptr.hit_descriptions.push(HitDescription {
                    word_info: word_index.clone(),
                    word_positions: word_pos[index_other]
                });
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


pub struct HitDescription<'a> {
    pub word_info: std::rc::Rc<(String,u64,u8,usize)>,
    pub word_positions: &'a [u32],
}

pub struct IndexHit<'a> {
    pub content: &'a str,
    pub hit_descriptions: Vec<HitDescription<'a>>,
}

impl<'a> IndexHit<'a> {
    pub fn sort_by_match_proximity(&mut self) {
        //Smaller is better!
        self.hit_descriptions.sort_by(|x,y|{y.word_info.2.cmp(&x.word_info.2)});
    }
}

pub enum PreviewBoundary {
    OuterWords,
    MaxDistance,
    SentenceBoundary,
    BoundedSentenceBoundary(usize),
    MaximizedSentenceBoundary(usize),
}

pub struct PreviewOptions<'a> {
    map_func: Box<dyn FnMut(&regex::Captures)->String+'a>,
    best_match: bool,
    allow_partial_match: bool,
    match_best_hits_only: bool,
    max_distance: usize,
    boundary: PreviewBoundary,
}

impl<'a> PreviewOptions<'a> {
    pub fn new() -> PreviewOptions<'a> {
        PreviewOptions {
            map_func: Box::new(|x|{String::from(x.get(0).unwrap().as_str())}),
            best_match: false,
            allow_partial_match: true,
            match_best_hits_only: true,
            max_distance: 10_000_000,
            boundary: PreviewBoundary::OuterWords,
        }
    }

    fn move_index_to_next_char_boundary(content: &str, mut index: usize) -> usize {
        while !content.is_char_boundary(index) {
            index+=1;
        }
        index
    }

    fn move_index_to_last_char_boundary(content: &str, mut index: usize) -> usize {
        while !content.is_char_boundary(index) {
            index-=1;
        }
        index
    }

    fn move_index_to_next_word_character(content: &str, index: usize, end: usize) -> usize {
        let regex = Regex::new(r"\w").unwrap();
        if let Some(first_word_start) = regex.find_at(&content[..end],index) {
            return first_word_start.start();
        }
        index
    }

    fn find_sentence_boundary(content: &str, start: usize, end: usize) -> Option<usize> {
        content[start..end].find(&['.','!','?'][..]).map(|x|{x+start})
    }

    fn rfind_sentence_boundary(content: &str, start: usize, end: usize) -> Option<usize> {
        content[start..end].rfind(&['.','!','?'][..]).map(|x|x+start)
    }

    fn find_sentence_boundary_max(content: &str, start: usize, end: usize) -> Option<usize> {
        if start == 0 {
            return None;
        }
        PreviewOptions::find_sentence_boundary(content, start, end)
    }

    fn rfind_sentence_boundary_max(content: &str, start: usize, end: usize) -> Option<usize> {
        if end == content.len() {
            return None;
        }
        PreviewOptions::rfind_sentence_boundary(content, start, end)
    }

    fn retrieve_preview<'b>(&self, content: &'b str, start: usize, end: usize) -> &'b str {
        match self.boundary {
            PreviewBoundary::OuterWords => &content[start as usize..end as usize],
            PreviewBoundary::SentenceBoundary => {
                let new_start = PreviewOptions::rfind_sentence_boundary(content, 0, start);
                let new_end = PreviewOptions::find_sentence_boundary(content, end, content.len());
                &content[new_start.map(|x| PreviewOptions::move_index_to_next_word_character(content, x, start)).unwrap_or(0)..new_end.map(|x|x+1).unwrap_or(content.len())]
            },
            PreviewBoundary::MaxDistance => {
                let dist = end - start;
                let mut new_start = start.checked_sub((self.max_distance-dist)/2).unwrap_or(0);
                let mut new_end = std::cmp::min(end.checked_add((self.max_distance-dist)/2).unwrap() as usize,content.len());
                new_start = PreviewOptions::move_index_to_next_char_boundary(content, new_start);
                new_end = PreviewOptions::move_index_to_last_char_boundary(content, new_end);
                &content[new_start as usize..new_end]
            },
            PreviewBoundary::BoundedSentenceBoundary(dist) => {
                let mut bounding_start = start.checked_sub(dist).unwrap_or(0);
                let mut bounding_end = std::cmp::min(end.checked_add(dist).unwrap() as usize,content.len());
                bounding_start = PreviewOptions::move_index_to_next_char_boundary(content, bounding_start);
                bounding_end = PreviewOptions::move_index_to_last_char_boundary(content, bounding_end);

                let new_start = PreviewOptions::rfind_sentence_boundary(content, bounding_start, start);
                let new_end = PreviewOptions::find_sentence_boundary(content, end, bounding_end);

                &content[new_start.map(|x| PreviewOptions::move_index_to_next_word_character(content, x, start)).unwrap_or(bounding_start)..new_end.map(|x|x+1).unwrap_or(bounding_end)]
            }
            PreviewBoundary::MaximizedSentenceBoundary(dist) => {
                let mut bounding_start = start.checked_sub(dist).unwrap_or(0);
                let mut bounding_end = std::cmp::min(end.checked_add(dist).unwrap() as usize,content.len());
                bounding_start = PreviewOptions::move_index_to_next_char_boundary(content, bounding_start);
                bounding_end = PreviewOptions::move_index_to_last_char_boundary(content, bounding_end);

                let new_start = PreviewOptions::find_sentence_boundary_max(content, bounding_start, start);
                let new_end = PreviewOptions::rfind_sentence_boundary_max(content, end, bounding_end);

                &content[new_start.map(|x| PreviewOptions::move_index_to_next_word_character(content, x, start)).unwrap_or(bounding_start as usize)..new_end.map(|x|x+1).unwrap_or(bounding_end)]
            }
        }
    }

    pub fn match_best_hits_only(mut self, match_best_hits_only: bool) -> Self {
        self.match_best_hits_only = match_best_hits_only;
        self
    }

    pub fn allow_partial_match(mut self, allow_partial_matches: bool) -> Self {
        self.allow_partial_match = allow_partial_matches;
        self
    }

    pub fn boundary(mut self, boundary: PreviewBoundary) -> Self {
        self.boundary = boundary;
        self
    }

    pub fn match_best(mut self, best: bool) -> Self {
        self.best_match = best;
        self
    }

    pub fn max_distance(mut self, value: usize) -> Self {
        self.max_distance = value;
        self
    }

    pub fn on_highlight<F: 'a+FnMut(&regex::Captures)->String>(mut self, func: F) -> Self {
        self.map_func = Box::new(func);
        self
    }
}

pub struct SearchHit<'a> {
    pub doc_ptr: u64,
    pub metadata: &'a [u8],
    pub doc_score: f32,
    pub index_hits: Vec<IndexHit<'a>>,
}

impl<'a> SearchHit<'a> {
    pub fn load_metadata<X: Serialize+Deserialize<'a>>(&self) -> Result<X,bincode::Error> {
        Ok(bincode::deserialize(self.metadata)?)
    }

    fn calculate_hit_score(hits: &[HeapMinDistance]) -> u32 {
        let mut score: u32 = hits.iter().map(|x| x.score).sum();
        score+=1000/hits.len() as u32;
        score
    }

    fn calculate_best_hit_score<'b>(second: &'b [HeapMinDistance<'a>],max_distance: u32, new_index: usize, best_score_until_now: u32, last_start_index: &mut u32) -> Option<(u32,Vec<&'a HitDescription<'a>>,u32,u32)> {
        let mut best_score = 0;
        let mut start = 0;
        let mut end = 0;
        let mut winning_vec: &[HeapMinDistance<'a>] = &[];

        for y in *last_start_index as usize..new_index {
            if (second[y].value-second[new_index].value)<=max_distance {
                let score = SearchHit::calculate_hit_score(&second[y..new_index+1]);
                if score > best_score_until_now {
                    best_score = score;
                    winning_vec = &second[y..new_index+1];
                    start = second[new_index].value;
                    end = second[y].value+second[y].hit.word_info.0.len() as u32;
                }
                *last_start_index = y.checked_sub(1).unwrap_or(0) as u32;
                break;
            }
        }

        if best_score > best_score_until_now {
            return Some((best_score,winning_vec.iter().map(|x|x.hit.clone()).collect(),start,end));
        }
        None
    }

    fn calculate_best_hit_score_full<'b>(second: &'b [HeapMinDistance<'a>],max_distance: u32) -> Option<(u32,Vec<&'a HitDescription<'a>>,u32,u32)> {
        let mut best_score = 0;
        let mut start = 0;
        let mut end = 0;
        let mut winning_vec: &[HeapMinDistance<'a>] = &[];

        for x in 0..second.len() {
            for y in x..second.len() {
                if (second[x].value-second[y].value)>max_distance {
                    let score = SearchHit::calculate_hit_score(&second[x..y]);
                    if score > best_score {
                        best_score = score;
                        winning_vec = &second[x..y];
                        start = second[y-1].value;
                        end = second[x].value+second[x].hit.word_info.0.len() as u32;
                    }
                    break;
                }
            }
        }

        Some((best_score,winning_vec.iter().map(|x|x.hit.clone()).collect(),start,end))
    }

    fn create_preview_on_hits<'options_livetime,'b>(content: &'b str, hits: Vec<&'a HitDescription>, options: &PreviewOptions<'options_livetime>) -> Option<(&'b str,Vec<&'a HitDescription<'a>>)> {
        let mut min_heap = std::collections::BinaryHeap::with_capacity(hits.len()+1);
        let mut indices = Vec::with_capacity(hits.len());
        let mut current_canidates = vec![std::u32::MAX;hits.len()];
        let mut sorted_canidates = Vec::with_capacity(hits.len());

        let mut min_d = std::u32::MAX;
        let mut max_d = 0;
        for x in 0..hits.len() {
            let mut iter = hits[x].word_positions.iter();
            let value = *iter.next().unwrap();
            if value < min_d {
                min_d = value;
            }
            if value > max_d {
                max_d = value+hits[x].word_info.0.len() as u32;
            }
            min_heap.push(HeapIndexValue {
                value: value,
                index: x
            });

            indices.push(iter);
        }

        sorted_canidates.sort_unstable();

        let mut min_dist = std::u32::MAX;
        let mut start = 0;

        let mut max_score = 0;
        let mut partial_start = 0;
        let mut partial_end = 0;
        let mut partial_matches = Vec::new();
        let mut last_start_index = 0;

        while let Some(min) = min_heap.pop() {
            let tmp_value = current_canidates[min.index];
            current_canidates[min.index] = min.value;
            if tmp_value == std::u32::MAX {
                sorted_canidates.push(HeapMinDistance {
                    value: min.value,
                    hit: hits[min.index],
                    score: 10_000/(hits[min.index].word_info.2 as u32+1)
                });
            }
            else if options.allow_partial_match {
                let index_old = sorted_canidates.binary_search_by(|x|x.value.cmp(&tmp_value).reverse()).unwrap();
                let mut old_value = sorted_canidates.remove(index_old);
                old_value.value = min.value;
                sorted_canidates.push(old_value);
                if index_old == 0 {
                    max_d = sorted_canidates[0].value+sorted_canidates[0].hit.word_info.0.len() as u32;
                }
            }
            else {
                if tmp_value == max_d-hits[min.index].word_info.0.len() as u32 {
                    max_d = *current_canidates.iter().max().unwrap();
                }
            }

            if min.value < min_d {
                min_d = min.value;
            }

            let distance = max_d-min_d;

            if distance < options.max_distance as u32 && sorted_canidates.len() == hits.len() {
                if !options.best_match {
                    return Some((options.retrieve_preview(content,min_d as usize,max_d as usize),hits));
                }
                if distance < min_dist {
                    min_dist = distance;
                    start = min_d;
                }
            }
            else if options.allow_partial_match {
                let result = if max_score == 0 {
                    SearchHit::calculate_best_hit_score_full(&sorted_canidates,options.max_distance as u32)
                }
                else {
                    SearchHit::calculate_best_hit_score(&sorted_canidates,options.max_distance as u32,sorted_canidates.len()-1,max_score,&mut last_start_index)
                };

                if let Some((score,winning_vec, x,y)) =  result {
                    max_score = score;
                    partial_matches = winning_vec;
                    partial_start = x;
                    partial_end = y;
                }
            }
            if let Some(value) = indices[min.index].next() {
                min_heap.push(HeapIndexValue  {
                    value: *value,
                    index: min.index
                });
            }
        }

        if min_dist > options.max_distance as u32 {
            if options.allow_partial_match {
                return Some((options.retrieve_preview(content,partial_start as usize,partial_end as usize),partial_matches))
            }
            else {
                return None;
            }
        }
        Some((options.retrieve_preview(content,start as usize,min_dist as usize+start as usize),hits))
    }

    pub fn create_preview<'options_livetime, T: Into<i32>>(&self, on_index: T, opts: Option<PreviewOptions<'options_livetime>>) -> Option<String> {
        let preview_options = opts.unwrap_or(PreviewOptions::new());

        let index = on_index.into() as usize;
        let mut checker: std::collections::HashMap<usize,&HitDescription<'a>> = std::collections::HashMap::new();
        for x in self.index_hits[index].hit_descriptions.iter() {
            if let Some(val) = checker.get(&x.word_info.3) {
                if val.word_info.2 > x.word_info.2 {
                    checker.insert(x.word_info.3,x);
                }
            }
            else {
                checker.insert(x.word_info.3,x);
            }
        }
        let preview = if preview_options.match_best_hits_only {
            let mut preview = Vec::with_capacity(checker.len());
            for (_,value) in checker.iter() {
                preview.push(*value);
            }
            preview
        }
        else {
            self.index_hits[index].hit_descriptions.iter().map(|x|x).collect()
        };
        

        let (prev,matched_words) = SearchHit::create_preview_on_hits(self.index_hits[index].content,preview,&preview_options)?;
        
        let mut string_builder = String::from(r"\b(");
        for x in matched_words.iter() {
            let escaped_string = regex::escape(&x.word_info.0);
            if string_builder.len() == 3 {
                string_builder = string_builder+&escaped_string;
            }
            else {
                string_builder = string_builder+"|"+&escaped_string;
            }
        }
        string_builder.push_str(r")\b");

        let reg = regex::RegexBuilder::new(&string_builder)
                .case_insensitive(true)
                .build()
                .expect("Invalid Regex");
        Some(reg.replace_all(prev,preview_options.map_func).to_string())
    }
}

pub struct DocumentSearchResult<'a> {
    pub hits: Vec<SearchHit<'a>>,
}


#[derive(Eq)]
pub struct HeapIndexValue {
    value: u32,
    index: usize,
}

impl std::cmp::PartialEq for HeapIndexValue {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl std::cmp::PartialOrd for HeapIndexValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

impl std::cmp::Ord for HeapIndexValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
       self.value.cmp(&other.value)
    }
}

pub struct HeapMinDistance<'a> {
    value: u32,
    hit: &'a HitDescription<'a>,
    score: u32,
}

impl<'a> std::cmp::PartialEq for HeapMinDistance<'a> {
    fn eq(&self, other: &Self) -> bool {
        other.value == self.value
    }
}

impl<'a> std::cmp::PartialOrd for HeapMinDistance<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(other.value.cmp(&self.value))
    }
}

impl<'a> Eq for HeapMinDistance<'a> {}

impl<'a> std::cmp::Ord for HeapMinDistance<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
       other.value.cmp(&self.value)
    }
}

impl<'a> DocumentSearchResult<'a> {
    pub fn sort_by_score(mut self) -> Self {
        self.hits.sort_by(|x,y|{x.doc_score.partial_cmp(&y.doc_score).unwrap_or(std::cmp::Ordering::Equal)});
        self
    }

    pub fn get<T: Into<i32>>(&self, index: T) -> &SearchHit<'a> {
        &self.hits[index.into() as usize]
    }

    pub fn start(mut self, start: usize) -> Self {
        if start >= self.hits.len() {
            self.hits.clear();
            return self;
        }
        self.hits.drain(0..start);
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        if self.hits.len() > limit {
            self.hits.drain(limit..);
        }
        self
    }

    pub fn and(mut self, mut other: DocumentSearchResult<'a>) -> DocumentSearchResult<'a> {
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
                    self.hits[index_own].index_hits[hit].hit_descriptions.append(&mut other.hits[index_other].index_hits[hit].hit_descriptions);
                }
                index_own+=1;
            }
            index_other+=1;
        }
    }

    pub fn or(mut self, mut other: DocumentSearchResult<'a>) -> DocumentSearchResult<'a> {
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
                    self.hits[index_own].index_hits[hit].hit_descriptions.append(&mut other.hits[index_other].index_hits[hit].hit_descriptions);
                }
                index_own+=1;
            }
            else {
                let new_hit = SearchHit {
                    doc_ptr: other.hits[index_other].doc_ptr,
                    doc_score: other.hits[index_other].doc_score,
                    metadata: other.hits[index_other].metadata,
                    index_hits: other.hits[index_other].index_hits.drain(..).collect(),
                };
                self.hits.insert(index_own, new_hit);
            }
            index_other+=1;
        }
    }

    pub fn boost_score_merge(mut self, mut other: DocumentSearchResult<'a>) -> DocumentSearchResult<'a> {
        let mut index_own = 0;
        let mut index_other = 0;

        loop {
            if index_own == self.hits.len() || index_other>=other.hits.len() {
                return self;
            }

            if self.hits[index_own].doc_ptr < other.hits[index_other].doc_ptr {
                index_own+=1;
                continue;
            }
            else if self.hits[index_own].doc_ptr == other.hits[index_other].doc_ptr {
                self.hits[index_own].doc_score += other.hits[index_other].doc_score;
                for hit in 0..other.hits[index_other].index_hits.len() {
                    self.hits[index_own].index_hits[hit].hit_descriptions.append(&mut other.hits[index_other].index_hits[hit].hit_descriptions);
                }
                index_own+=1;
            }
            
            index_other+=1;
        }
    }

    pub fn not(mut self, other: DocumentSearchResult<'a>) -> DocumentSearchResult<'a> {
        let mut index_own = 0;
        let mut index_other = 0;

        loop {
            if index_own == self.hits.len() || index_other>=other.hits.len(){
                return self;
            }

            if self.hits[index_own].doc_ptr < other.hits[index_other].doc_ptr {
                index_own+=1;
                continue;
            }
            else if self.hits[index_own].doc_ptr == other.hits[index_other].doc_ptr {
                self.hits.remove(index_own);
            }
            index_other+=1;
        }
    }
}

pub struct IndexReader<'a> {
    data: &'a [u8],
    index_data: std::collections::HashMap<i32,(&'a [u8], &'a [u8])>,
    query_id: std::sync::atomic::AtomicUsize,
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
            query_id: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn search(&'a self, query: Query<'a>) -> Result<DocumentSearchResult<'a>,std::io::Error> {
        self.search_query(query)
    }

    pub fn search_query(&'a self, mut query: Query<'a>) -> Result<DocumentSearchResult<'a>,std::io::Error> {
        query.query_id = self.query_id.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
        let mut result: DocumentSearchResult<'a> = DocumentSearchResult {
            hits: Vec::new()
        };
        
        if query.target_index.len() == 0 {
            query.target_index = self.index_data.keys().map(|x| (1.0,*x)).collect();
        }

        for index in query.target_index.iter() {
            if let Some((index_automaton,index_data)) = self.index_data.get(&index.1) {
                result = result.or(match query.query_settings {
                    QueryOperationSettings::Exact => {
                        self.load_hits(
                            self.do_search_in_index(fst::automaton::Str::new(query.query),index_automaton,query.query_id)?,
                            index.1,
                            index_data,
                            query.boost_factor*index.0
                        )
                    },
                    QueryOperationSettings::LevenstheinDistance1 => {
                        self.load_hits(
                            self.do_search_in_index_levensthein(query.query,1,index_automaton,query.query_id)?,
                            index.1,
                            index_data,
                            query.boost_factor*index.0
                        )
                    },
                    QueryOperationSettings::LevenstheinDistance2 => {
                        self.load_hits(
                            self.do_search_in_index_levensthein(query.query,2,index_automaton,query.query_id)?,
                            index.1,
                            index_data,
                            query.boost_factor*index.0
                        )
                    },
                    QueryOperationSettings::StartsWith => {
                        self.load_hits(
                            self.do_search_in_index(fst::automaton::Str::new(query.query).starts_with(),index_automaton,query.query_id)?,
                            index.1,
                            index_data,
                            query.boost_factor*index.0
                        )
                    },
                    QueryOperationSettings::Subsequence => {
                        self.load_hits(
                            self.do_search_in_index(fst::automaton::Subsequence::new(query.query).starts_with(),index_automaton,query.query_id)?,
                            index.1,
                            index_data,
                            query.boost_factor*index.0
                        )
                    }
                }?);
            }
            else {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound,format!("Could not find index or index is not loaded \"{:?}\"",index)));
            }
        }
        Ok(result)
    }

    fn do_search_in_index_levensthein(&'a self,  query: &'a str, distance: u8, index_automaton: &'a [u8], query_id: usize) -> Result<Vec<std::rc::Rc<(String,u64,u8,usize)>>,std::io::Error> {
        let load_automaton = fst::Map::new(index_automaton).unwrap();
        let auto = if distance == 1 {
            DFAWrapper(AUTOMATON_DIST_1.build_dfa(query))
        }
        else {
            DFAWrapper(AUTOMATON_DIST_2.build_dfa(query))
        };

        let mut result = load_automaton.search_with_state(&auto).into_stream();
        
        let mut sorted_states:Vec<std::rc::Rc<(String,u64,u8,usize)>> = Vec::with_capacity(100);
        while let Some((key_u8,value,state)) = result.next() {
            let key = unsafe{std::str::from_utf8_unchecked(key_u8).to_string()};
            match auto.0.distance(state) {
                Distance::Exact(a) => {sorted_states.push(std::rc::Rc::new((key,value,a,query_id)))},
                _ => {}
            }
        }
        sorted_states.sort_by(|x,y| {x.2.cmp(&y.2)});
        Ok(sorted_states)
    }

    fn do_search_in_index<A: fst::Automaton>(&'a self, automaton: A,index_automaton: &'a [u8], query_id: usize) -> Result<Vec<std::rc::Rc<(String,u64,u8,usize)>>,std::io::Error> {
        let load_automaton = fst::Map::new(index_automaton).unwrap();

        let mut return_value = Vec::with_capacity(100);
        let mut result = load_automaton.search(automaton).into_stream();
        while let Some((key_u8,value)) = result.next() {
            let key = unsafe{std::str::from_utf8_unchecked(key_u8).to_string()};
            return_value.push(std::rc::Rc::new((key,value,0,query_id)));
        }
        Ok(return_value)
    }

    pub fn load_hits(&'a self, matches: Vec<std::rc::Rc<(String,u64,u8,usize)>>, index: i32, val: &'a [u8], boost_factor: f32) -> Result<DocumentSearchResult<'a>,std::io::Error> 
        {
        let mut orderedcoll: OrderedCollector<'a> = OrderedCollector::new(index, boost_factor);

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

    pub fn get_suggestions<T: Into<i32>>(&'a self, index: T, query: &str, limit: usize) -> Result<Vec<String>,std::io::Error> {
        let mut return_value = if limit < 1000 {Vec::with_capacity(limit)} else {Vec::with_capacity(limit)};
        let i32_index: i32 = index.into();
        if let Some((index_automaton,_)) = self.index_data.get(&i32_index) {
            let load_automaton = fst::Map::new(index_automaton).unwrap();

            let mut result = load_automaton.search(fst::automaton::Str::new(query).starts_with()).into_stream();
            while let Some((key_u8,_)) = result.next() {
                let key = unsafe{std::str::from_utf8_unchecked(key_u8).to_string()};
                return_value.push(key);
                if return_value.len() >= limit {
                    return Ok(return_value);
                }
            }
        }
        else {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound,format!("Could not find index or index is not loaded \"{}\"",i32_index)));
        }
        Ok(return_value)
    }
}

#[cfg(test)]
mod tests {
    use super::{Document,RAMFilesystem,Filesystem,Index,IndexWriter,Query,MMapedFilesystem,IndexReader,RawTokenizer,Tokenizer,filter_long,PreviewOptions,PreviewBoundary,HeapIndexValue,DocumentTermCollection};
    use membuffer::MemBufferSerialize;
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
        ramdir.write_file("fuchs", &[0,10,0,100,100]).unwrap();
        assert_eq!(ramdir.load_file("fuchs").unwrap(),[0,10,0,100,100]);
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
    pub fn check_writer_real_mmaped() {
        let _ = std::fs::remove_dir_all("second");
        std::fs::create_dir("second").unwrap();
        let mut mmaped = MMapedFilesystem::from("second").unwrap();
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
            new_doc.add_field(IndexEnum::Body, "hello how are you?");
            index.add_document(new_doc);
            let val = index.commit();
            val.unwrap();
            //assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
            assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,1);
            assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,1);
            assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),4);
            assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),4);
        }

        let mut ram = MMapedFilesystem::from("second").unwrap();
        let second_load = IndexWriter::from_fs(&mut ram).unwrap();
        assert_eq!(second_load.index_locked,true);
        assert_eq!(second_load.indexes[IndexEnum::Title as usize].total_indexed_count,1);
        assert_eq!(second_load.indexes[IndexEnum::Body as usize].total_indexed_count,1);
        assert_eq!(second_load.indexes[IndexEnum::Body as usize].term_index.len(),4);
        assert_eq!(second_load.indexes[IndexEnum::Title as usize].term_index.len(),4);

    }

    #[test]
    pub fn check_writer_real() {
        let _ = std::fs::remove_dir_all("fuchs");
        std::fs::create_dir("fuchs").unwrap();
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
            let val = index.commit();
            val.unwrap();
            //assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
            assert_eq!(index.indexes[IndexEnum::Title as usize].total_indexed_count,1);
            assert_eq!(index.indexes[IndexEnum::Body as usize].total_indexed_count,0);
            assert_eq!(index.indexes[IndexEnum::Body as usize].term_index.len(),0);
            assert_eq!(index.indexes[IndexEnum::Title as usize].term_index.len(),4);
        }

        mmaped.persist("fuchs").unwrap();

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
            let id0 = index.add_document(new_doc0);


            let mut new_doc = Document::new(&new_meta);
            new_doc.add_field(IndexEnum::Title, "hello how are you? Or hallo seems equally good. hollo");
            new_doc.add_field(IndexEnum::Body, "/alex/nice/");
            let id1 = index.add_document(new_doc);
            assert_ne!(id0,id1);

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
            let mut result = reader.search(query).unwrap();
            let mut result2 = reader.search(query2).unwrap();
            result = result.sort_by_score();
            result2 = result2.sort_by_score();
            assert_eq!(result.hits[0].doc_score*2.0, result2.hits[0].doc_score);
        }
        
        {
            let query = Query::fuzzy("hello",1).target(IndexEnum::Title);
            let mut result = reader.search(query).unwrap();
            result = result.sort_by_score();
            assert_eq!(result.hits.len(),3);
        }

        {
            let query = Query::fuzzy("hello",2).target(IndexEnum::Title);
            let mut result = reader.search(query).unwrap();
            result = result.sort_by_score();
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

        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title,IndexEnum::Body]).unwrap();
        
        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::starts_with("/alex/").target(IndexEnum::Body);
            let result = reader.search(query).unwrap();
            let result2 = reader.search(query_path).unwrap();
            let union = result.and(result2);
            assert_eq!(union.hits.len(), 1);
        }

        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::starts_with("/alex/").target(IndexEnum::Body);
            let result = reader.search(query).unwrap();
            let result2 = reader.search(query_path).unwrap();
            let or = result.or(result2);
            assert_eq!(or.hits.len(),2);
        }

        {
            let query = Query::fuzzy("hello",1).target(IndexEnum::Title);
            let query_path = Query::fuzzy("/alex/",1).target(IndexEnum::Body);
            let result = reader.search(query).unwrap();
            let result2 = reader.search(query_path).unwrap();
            let or = result.or(result2);
            assert_eq!(or.hits.len(),2);
        }

        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::exact("sad").target(IndexEnum::Title);
            let result = reader.search(query).unwrap();
            let result2 = reader.search(query_path).unwrap();
            let or = result2.or(result);
            assert_eq!(or.hits.len(),2);
        }

        {
            let query = Query::exact("hello").target(IndexEnum::Title);
            let query_path = Query::exact("sad").target(IndexEnum::Title);
            let result = reader.search(query).unwrap();
            let result2 = reader.search(query_path).unwrap();
            let _or = result.and(result2);
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
        let result = reader.search(query).unwrap();
        let meta:DocumentMeta = result.get(0).load_metadata().unwrap();
        assert_eq!(meta.path,"main.txt");
        assert_eq!(meta.title,"Intel developers system manual");

        assert_eq!(result.hits.len(), 2);
        assert_eq!(result.limit(1).hits.len(), 1);


        
        let query2 = Query::exact("hello").target(IndexEnum::Body);
        let result2 = reader.search(query2);
        assert_eq!(result2.is_err(),true);
    }


    #[test]
    pub fn check_preview() {
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
            new_doc2.add_field(IndexEnum::Title, "This is a sad title with hello in it! For the sake of it hello sad.");
            index.add_document(new_doc2);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
        }

        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title]).unwrap();
        let query = Query::exact("hello").target(IndexEnum::Title);
        let result = reader.search(query).unwrap();
        assert_eq!(result.hits.len(), 2);
        
        let query2 = Query::exact("sad").target(IndexEnum::Title);
        let result2 = reader.search(query2);
        assert_eq!(result2.is_err(),false);

        let query3 = Query::exact("it").target(IndexEnum::Title);
        let result3 = reader.search(query3);

        let combined = result.and(result2.unwrap()).and(result3.unwrap()).sort_by_score();
        {
            let preview_options = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::BoundedSentenceBoundary(100));
            let preview = combined.hits[0].create_preview(IndexEnum::Title,Some(preview_options)).unwrap();
            assert_eq!("For the sake of |it| |hello| |sad|.",preview);
        }

        {
            let preview_options_word = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::OuterWords);
            let preview_words = combined.hits[0].create_preview(IndexEnum::Title,Some(preview_options_word)).unwrap();
            assert_eq!("|it| |hello| |sad|",preview_words);
        }

        {
            let preview_options_word = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::OuterWords).allow_partial_match(true).max_distance(6);
            let preview_words = combined.hits[0].create_preview(IndexEnum::Title,Some(preview_options_word)).unwrap();
            assert_eq!("|hello| |sad|",preview_words);
        }

        {
            let preview_options_maximised = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::MaximizedSentenceBoundary(100));
            let preview_words_max = combined.hits[0].create_preview(IndexEnum::Title,Some(preview_options_maximised)).unwrap();
            assert_eq!("This is a |sad| title with |hello| in |it|! For the sake of |it| |hello| |sad|.",preview_words_max);
        }

        {
            let preview_options_maximised = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::MaximizedSentenceBoundary(30));
            let preview_words_max = combined.hits[0].create_preview(IndexEnum::Title,Some(preview_options_maximised)).unwrap();
            assert_eq!("For the sake of |it| |hello| |sad|.",preview_words_max);
        }

        {
            let query = Query::exact("sake").target(IndexEnum::Title);
            let result = reader.search(query).unwrap();
            let preview_options = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::BoundedSentenceBoundary(100));
            let preview = result.hits[0].create_preview(IndexEnum::Title,Some(preview_options)).unwrap();
            assert_eq!("For the |sake| of it hello sad.",preview);

        }

        {
            let query = Query::exact("sake").target(IndexEnum::Title);
            let result = reader.search(query).unwrap();
            let preview_options = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::MaxDistance).max_distance(20);
            let preview = result.hits[0].create_preview(IndexEnum::Title,Some(preview_options)).unwrap();
            assert_eq!("For the |sake| of it h",preview);
        }

        {
            let query = Query::exact("sake").target(IndexEnum::Title);
            let result = reader.search(query).unwrap();
            let preview_options = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::SentenceBoundary).max_distance(20);
            let preview = result.hits[0].create_preview(IndexEnum::Title,Some(preview_options)).unwrap();
            assert_eq!("For the |sake| of it hello sad.",preview);

        }

        {
            //Check matching the highest proximity match as the query will also match "in" and "is"
            let query = Query::fuzzy("it",1).target(IndexEnum::Title);
            let result = reader.search(query).unwrap();
            let preview_options = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::SentenceBoundary).max_distance(20);
            let preview = result.hits[0].create_preview(IndexEnum::Title,Some(preview_options)).unwrap();
            assert_eq!("For the sake of |it| hello sad.",preview);

        }
    }

    #[test]
    pub fn check_preview_additional() {
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
            new_doc.add_field(IndexEnum::Title, "Alexander Leonhardt: Generators/Research/Blog about theoretical computer science");
            index.add_document(new_doc);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
        }

        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title]).unwrap();
        let query = Query::starts_with("a").target(IndexEnum::Title);
        let result = reader.search(query).unwrap();
        
        {
            let preview_options = PreviewOptions::new().match_best(false).match_best_hits_only(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::BoundedSentenceBoundary(100));
            let preview = result.hits[0].create_preview(IndexEnum::Title,Some(preview_options)).unwrap();
            assert_eq!("|Alexander| Leonhardt: Generators/Research/Blog |about| theoretical computer science",preview);
        }
    }

    #[test]
    pub fn check_start_query_at() {
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
            new_doc2.add_field(IndexEnum::Title, "This is a sad title with hello in it! For the sake of it hello sad.");
            index.add_document(new_doc2);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
        }

        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title]).unwrap();
        let query = Query::exact("hello").target(IndexEnum::Title);
        let result = reader.search(query).unwrap();
        assert_eq!(result.hits.len(), 2);
        let _doc_ptr1 = result.hits[0].doc_ptr;
        let doc_ptr2 = result.hits[1].doc_ptr;
        let new_result = result.start(1);
        assert_eq!(doc_ptr2,new_result.hits[0].doc_ptr);
        assert_eq!(doc_ptr2,new_result.get(0).doc_ptr);
        let second_new = new_result.start(5);
        assert_eq!(second_new.hits.len(),0);
    }

    #[test]
    pub fn check_start_streaming_heap_generation() {
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
            new_doc2.add_field(IndexEnum::Title, "This is a sad title with hello in it! For the sake of it hello sad. At least 3 occurences of hello are needed, hello!");
            index.add_document(new_doc2);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
        }

        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title]).unwrap();
        let query = Query::exact("hello").target(IndexEnum::Title);
        let result = reader.search(query).unwrap().sort_by_score();
        assert_eq!(result.hits.len(), 2);
        
        let preview_options = PreviewOptions::new().match_best(true).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::MaxDistance).max_distance(20);
        let preview = result.hits[0].create_preview(IndexEnum::Title,Some(preview_options)).unwrap();
        assert_eq!("eeded, |hello|!",preview);
    }

    #[test]
    pub fn check_heap_index_cmp() {
        let val = HeapIndexValue{value: 10, index: 0};
        let other = HeapIndexValue{value: 11, index: 1};
        assert_eq!(val<other,true);
        assert_eq!(val==other, false);
        assert_eq!(val.cmp(&other),std::cmp::Ordering::Less);
        assert_eq!(DocumentTermCollection::get_mem_buffer_type(),10);
    }

    #[test]
    pub fn check_suggestions() {
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
            new_doc2.add_field(IndexEnum::Title, "This is a sad title with hello in it! For the sake of it hello sad. How are you?");
            index.add_document(new_doc2);

            let val = index.commit();
            assert_eq!(val.is_err(),false);
            assert_eq!(index.index_locked,true);
        }

        
        let reader = IndexReader::from_fs(&mmaped,vec![IndexEnum::Title]).unwrap();
        
        {
            let result = reader.get_suggestions(IndexEnum::Title, "h", 1);
            assert_eq!(result.is_err(),false);
            let result_unwrappred = result.unwrap();
            assert_eq!(result_unwrappred.len(), 1);
            assert_eq!(result_unwrappred, vec!["hello"]);
        }
        {
            let result = reader.get_suggestions(IndexEnum::Title, "h", 10000);
            assert_eq!(result.is_err(),false);
            let result_unwrappred = result.unwrap();
            assert_eq!(result_unwrappred.len(), 2);
            assert_eq!(result_unwrappred, vec!["hello","how"]);
        }
        {
            let result = reader.get_suggestions(IndexEnum::Body, "h", 10000);
            assert_eq!(result.is_err(),true);
        }
    }
}


#[cfg(feature="bench")]
mod bench {
    use test::Bencher;
    use super::{Document,IndexWriter,Query,MMapedFilesystem,IndexReader,RAMFilesystem,PreviewBoundary,PreviewOptions};
    use serde::{Serialize,Deserialize};

    #[derive(Serialize,Deserialize)]
    struct DocumentMeta<'a> {
        title: &'a str,
        path: &'a str,
    }

    #[bench]
    fn check_reading_from_ram(b: &mut Bencher) {
        let mut mmaped = RAMFilesystem::new();

        {
            let mut new_writer = IndexWriter::from_fs(&mut mmaped).unwrap();
            new_writer.add_index(0);

            let hugestring = std::fs::read_to_string("main.txt").unwrap();

            let doc = DocumentMeta {
                title: "The modern Intel System Environment",
                path: "main.txt"
            };

            for _ in 0..10 {
                let mut document = Document::new(&doc);
                document.add_field(0, &hugestring);
                new_writer.add_document(document);
            }

            for _ in 0..10_000 {
                let mut document = Document::new(&doc);
                document.add_field(0,"and is indeed very nice write a small text about the live of brian to check how the time changes if i use a lot more text which should lead to more cache misses in theory");
                new_writer.add_document(document);
            }

            new_writer.commit().unwrap();
        }

        b.iter(|| {
            let reader = IndexReader::from_fs(&mmaped,vec![0]).unwrap();
            let result = reader.search(Query::fuzzy("and",1).target(0)).unwrap();
            assert_eq!(result.hits.len(),10_010);
            let mut _score = 0.0;
            for x in result.hits.iter() {
                _score+=x.doc_score;
            }
        });
    }

    #[bench]
    fn benchmark_mmap_performance(b: &mut Bencher)
    {
        let mut value = 0;
        b.iter(|| {
            let read_system = MMapedFilesystem::from("bench_data").unwrap();
            value += read_system.mapped_files.len();
        });
    }

    #[bench]
    fn check_reading_mmaped_directory(b: &mut Bencher)
    {
        b.iter(|| {
            let read_system = MMapedFilesystem::from("bench_data").unwrap();
            let reader = IndexReader::from_fs(&read_system,vec![0]).unwrap();
            let result = reader.search(Query::fuzzy("and",1).target(0)).unwrap();
            assert_eq!(result.hits.len(),10_010);
            let mut _score = 0.0;
            for x in result.hits.iter() {
                _score+=x.doc_score;
            }
        });
    }

    #[bench]
    fn check_creating_multi_preview(b: &mut Bencher) 
    {
        let read_system = MMapedFilesystem::from("bench_data").unwrap();
        let reader = IndexReader::from_fs(&read_system,vec![0]).unwrap();
        let result = reader.search(Query::fuzzy("and",1).target(0)).unwrap();
        let result2 = reader.search(Query::exact("write").target(0)).unwrap();
        let result3 = reader.search(Query::exact("how").target(0)).unwrap();
        let result4 = reader.search(Query::fuzzy("intel", 1).target(0)).unwrap();
        let end_result = result.and(result2).and(result3).and(result4).sort_by_score().limit(10);
        let mut result_len = 0;

        b.iter(|| {
            let preview_options = PreviewOptions::new().match_best(true).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::OuterWords).allow_partial_match(true).max_distance(200).match_best_hits_only(false);
            if let Some(preview) = end_result.hits[0].create_preview(0,Some(preview_options)) {
                result_len+=preview.len();
            }
        });
    }

    #[bench]
    fn check_creating_single_preview(b: &mut Bencher) 
    {
        let read_system = MMapedFilesystem::from("bench_data").unwrap();
        let reader = IndexReader::from_fs(&read_system,vec![0]).unwrap();
        let result2 = reader.search(Query::exact("write").target(0)).unwrap();
        let end_result = result2.sort_by_score().limit(10);
        let mut result_len = 0;

        b.iter(|| {
            let preview_options = PreviewOptions::new().match_best(false).on_highlight(|x|String::from("|")+x.get(0).unwrap().as_str()+"|").boundary(PreviewBoundary::OuterWords).max_distance(200).match_best_hits_only(false);
            if let Some(preview) = end_result.hits[0].create_preview(0,Some(preview_options)) {
                result_len+=preview.len();
            }
        });
    }

    #[bench]
    fn check_suggestion_speed(b: &mut Bencher)
    {
        let mut total = 0;
        b.iter(|| {
            let read_system = MMapedFilesystem::from("bench_data").unwrap();
            let reader = IndexReader::from_fs(&read_system,vec![0]).unwrap();
            let result = reader.get_suggestions(0, "an", 10).unwrap();
            assert_eq!(result.len(),10);
            total+=result.len();
        });
    }
}
