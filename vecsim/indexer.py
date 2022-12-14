from model import get_tokenizer, get_model
import re
import string
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
import redis.asyncio as redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField
from typing import Optional, Pattern

import config

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

       
    def read(self, sample_frac=None, limit=None):
        df = self._read_paper_df()
        df.reset_index(drop=True, inplace=True)
        if sample_frac:
            df = df.sample(frac=sample_frac)
        docs = df.input.to_list()
        if limit:
            docs = docs[:limit]
        return docs

class ColBERTFormator:    
    def __init__(self, doc_prefix, vec_prefix):
        self.doc_prefix = doc_prefix
        self.vec_prefix = vec_prefix

    def get_doc_key(self, key):
        return f"{self.doc_prefix}:{key}"
    def get_vec_key(self, key):
        return f"{self.vec_prefix}:{key}"

    def create_doc_formatting(self, doc_id,vecs, data):
        vecs = vecs.astype(np.float32)
        vecs_bytes = vecs.tobytes()
        docs = []        
        key = self.get_doc_key(f'{doc_id}')        
        doc = {
            'id': key,
            'doc_id': str(doc_id),
            'doc': data,
            'vector_matrix': vecs_bytes
        }
        docs.append(doc)
        return docs
    def create_vec_formatting(self, doc_id, vecs):        
        docs = []
        for vec_id, vec in enumerate(vecs):
            key = self.get_vec_key(f'{doc_id}-{vec_id}')
            vec_bytes = vec.tobytes()
            doc = {
                'id': key,
                'doc_id': str(doc_id),
                'vec_id': str(vec_id),
                'vector': vec_bytes
            }
            docs.append(doc)
        return docs
    def get_fields(self):
        block_size = 1078
        vector_field = VectorField(
            "vector",
            "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": 768,
                "DISTANCE_METRIC": "IP",
                "INITIAL_CAP": block_size,
                "BLOCK_SIZE": block_size
            }
        )
        doc_id = TagField("doc_id")
        vec_id = TagField("vec_id")
        # doc = TagField("doc")
        fields = [
            vector_field,
            doc_id,
            vec_id,
            doc
        ]
        return fields
    
class RedisIndexer():
    def __init__(self, index_name, host="redis", port="6379", url=None, n=50):
        self.redis_conn=None        
        self.semaphore = asyncio.Semaphore(n)
        self.url = url
        self.host = host
        self.port = port
        self.index_name=index_name

    async def init_redis(self):
        if self.redis_conn is None:
            if self.url is not None:
                self.redis_conn = await redis.from_url(self.url)
            else:
                self.redis_conn = await redis.Redis(host=self.host, port=self.port)
            
    async def write(self, data):
        if type(data) != list:            
            data = [data]
        for data_elem in tqdm(data):
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
        
# def preprocess(df):
#     df['input'] = df.apply(lambda r: r.title + r.abstract, axis=1)
#     df.reset_index(drop=True, inplace=True)
#     return df

def clean_description(description: str):
    if not description:
        return ""
    # remove unicode characters
    description = description.encode('ascii', 'ignore').decode()

    # remove punctuation
    description = re.sub('[%s]' % re.escape(string.punctuation), ' ', description)

    # clean up the spacing
    description = re.sub('\s{2,}', " ", description)

    # remove urls
    #description = re.sub("https*\S+", " ", description)

    # remove newlines
    description = description.replace("\n", " ")

    # remove all numbers
    #description = re.sub('\w*\d+\w*', '', description)

    # split on capitalized words
    description = " ".join(re.split('(?=[A-Z])', description))

    # clean up the spacing again
    description = re.sub('\s{2,}', " ", description)

    # make all words lowercase
    description = description.lower()

    return description

def preprocess(r):
    return clean_description(r.title + r.abstract)

async def main(filename):
    docs_source = DocumentSource(filename)
    indexer = RedisIndexer(index_name="doc", url=config.REDIS_URL)
    await indexer.init_redis()
    
    model = ColBERTModel()
    formator = ColBERTFormator(prefix="doc")
    
    for i,data in enumerate(docs_source.read(sample_frac=0.1)):
        # print(f"{i} data: {data}")
        data = preprocess(data)
        embeddings = model.compute_document_representation(data)
        # print(f"{i} data: {embeddings}")
        docs = formator.create_formating(i, embeddings, data)
        # print(docs)
        await indexer.write(docs)
    await indexer.print_dbsize()

    await indexer.create_index(
        *formator.get_fields(),
        prefix="doc"
    )
    print(f"{indexer.index_name} index created")

if __name__ == '__main__':
    
    filename = "../arxiv_papers_df.pkl"
    asyncio.run(
        main(filename)
    )
        
    

