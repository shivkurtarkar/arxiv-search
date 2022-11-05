from model import get_tokenizer, get_model
##
import asyncio

from tqdm import tqdm

import numpy as np
import pandas as pd
import pickle
from model import ColBERTModel

from redis.commands.search.field import (
    TagField,
    VectorField,
    NumericField,
    TextField
)
from redis.asyncio import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField
from typing import Optional, Pattern

# read docs
# preprocess
# add to redis
# create index

class DocumentSource:
    def __init__(self, filename):
        self.filename = filename
    
    def _read_paper_df(self) -> pd.DataFrame:
        with open(self.filename, "rb") as file:
            df = pickle.load(file)
        return df

    def _preprocess(self, df):
        df['input'] = df.apply(lambda r: r.title + r.abstract, axis=1)
        df.reset_index(drop=True, inplace=True)
        return df
        
    def read(self, sample_frac=None, limit=None):
        df = self._read_paper_df()
        df = self._preprocess(df)
        if sample_frac:
            df = df.sample(frac=sample_frac)
        docs = df.input.to_list()
        if limit:
            docs = docs[:limit]
        return docs

class ColBERTTSBoardFormator:
    def __init__(self, prefix):
        self.prefix = prefix

    def get_key(self, key):
        return f"{self.prefix}:{key}"    

    def create_formating(self, doc_id, vecs, data, tokens):
        vecs = vecs.astype(np.float32)
        vecs_bytes = vecs.tolist()
        docs = []        
        for vec_id, (vec, token) in enumerate(zip(vecs, tokens)):
            key = self.get_key(f'{doc_id}-{vec_id}')
            vec_bytes = vec.tolist()
            
            doc = {
                
                'id': key,
                'doc_id': str(doc_id),
                'vec_id': str(vec_id),
                'token': token,                
                'doc': data,
                'vector': vec_bytes,
                'vector_matrix': vecs_bytes,
                
            }
            docs.append(doc)
        return docs
    def get_fields(self):
        return ['doc_id', 'vec_id', 'token']
    
class RedisIndexer():
    def __init__(self, index_name, host="redis", port="6379", n=50):
        self.redis_conn=None        
        self.semaphore = asyncio.Semaphore(n)
        self.host = host
        self.port = port
        self.index_name=index_name

    async def init(self):
        if self.redis_conn is None:
            self.redis_conn = await Redis(host=self.host, port=self.port)
            
    async def write(self, data):
        if type(data) != list:            
            data = [data]
        for data_elem in data:
            # print(data_elem)
            await self.__write_to_redis(data_elem['id'], data_elem)

    async def __write_to_redis(self, key:str, data):
        async with self.semaphore:
            await self.redis_conn.hset(key, mapping=data)
    
    async def print_dbsize(self):
        print(await self.redis_conn.dbsize()) 
        
    async def create_index(
        self,
        *fields,
        prefix: str,
        overwrite=True
    ):
        # Create Index
        if overwrite:
            await self.redis_conn.ft(self.index_name).dropindex(delete_documents=False)
        await self.redis_conn.ft(self.index_name).create_index(
            fields = fields,
            definition= IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
        )

import os
class TensorBoardVisualizerIndexer:
    def __init__(self, index_name, log_dir='./logs/'):
        self.index_name = index_name
        self.log_dir = log_dir
        self.vecs = []
        self.labels = []
        
    def init(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)            
        
    def write(self, datas):
        for data in datas:
            # print(data['vector'])
            self.vecs.append(data['vector'])
            self.labels.append(data)
    def print_dbsize(self):
        print(len(self.vecs))
    def create_index(self, fields, i):
        self.vector_file_name_tsv = f"{self.log_dir}/{self.index_name}_vectors_{i}.tsv"
        self.label_file_name = f"{self.log_dir}/{self.index_name}_labels_{i}.tsv"
        self.vector_file_name = f"{self.log_dir}/{self.index_name}_vectors_{i}.txt"
        
        feature_list = np.asarray(self.vecs)
        print(f"feature_list: {len(feature_list)}")
        np.savetxt(self.vector_file_name,feature_list)
            
        with open(self.vector_file_name_tsv, 'w') as vector_file:
            for vector in self.vecs:
                formatted_vec = self._format([str(each) for each in vector])
                vector_file.write(f"{formatted_vec}\n")
        with open(self.label_file_name, 'w') as label_file:
            label_file.write(f"{self._format(fields)}\n")
            for labels in self.labels:
                label_list = []
                for field in fields:
                    label_list.append(labels[field].strip('\n').strip('\\'))
                formatted_labels = self._format(label_list)
                label_file.write(f"{formatted_labels}\n")
        self.vecs = []
        self.labels=[]

    def _format(self,vector):
        # if len(vector.shape)==2:
        #     vector = vector.squeeze(0)
        # elif len(vector.shape)>2:
        #     raise Exception('cant have more than 2 axis')
        serialized_vec = '\t'.join(vector)
        return serialized_vec
            
        
    
        
async def main(filename):
    docs_source = DocumentSource(filename)
    # indexer = RedisIndexer(index_name="doc")
    indexer = TensorBoardVisualizerIndexer(index_name="doc")
    indexer.init()
    
    model = ColBERTModel()
    formator = ColBERTTSBoardFormator(prefix="doc")
    
    j =0
    batch_size = 100
    for i,data in enumerate(tqdm(docs_source.read(sample_frac=0.1))):
        try:
            # print(f"{i} data: {data}")
            embeddings = model.compute_document_representation(data)
            # print(f"{i} data: {embeddings}")
            tokens = model.get_tokens(data)
            docs = formator.create_formating(i, embeddings, data, tokens)
            # print(docs)
        except Exception as e:
            print(f"exception occured at {i}")
        indexer.write(docs)
        if i%batch_size==0:
            indexer.create_index(formator.get_fields(), j)
            j+=1
    if len(indexer.vecs) > 0:
        indexer.create_index(formator.get_fields(), j)        
    indexer.print_dbsize()

    
    
    print(f"{indexer.index_name} index created")

if __name__ == '__main__':
    
    filename = "../arxiv_papers_df.pkl"
    asyncio.run(
        main(filename)
    )
        
    

