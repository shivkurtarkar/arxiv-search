import re

from redis.asyncio import Redis
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField
from typing import Optional, Pattern



##
import numpy as np
import asyncio
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


#  Query preprocessing
#  Search n vectors
#  Get docs
#  Compute Scores and rerank


#  Explainability
#  Query-doc vecs compute
#  Dotproduct
#  convert to image and plot

class RedisIndexer():
    def __init__(self, index_name, host="redis", port="6379", n=50):
        self.redis_conn=None        
        self.semaphore = asyncio.Semaphore(n)
        self.host = host
        self.port = port
        self.index_name=index_name

    async def init_redis(self):
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
        prefix: str
    ):
        # Create Index
        await self.redis_conn.ft(self.index_name).create_index(
            fields = fields,
            definition= IndexDefinition(prefix=[prefix], index_type=IndexType.HASH)
        )
    def vector_query(
        self,
        search_type: str="KNN",
        number_of_results: int=20
    ) -> Query:
        """
        Create a RediSearch query to perform hybrid vector and tag based searches.

        Args:
            search_type (str, optional): Style of search. Defaults to "KNN".
            number_of_results (int, optional): How many results to fetch. Defaults to 20.
        Returns:
            Query: RediSearch Query
        """
        # Parse tags to create query
        tag_query = "*"
        base_query = f'{tag_query}=>[{search_type} {number_of_results} @vector $vec_param AS vector_score]'
        return Query(base_query)\
            .sort_by("vector_score")\
            .paging(0, number_of_results)\
            .return_fields("doc_id", "doc", "vector_matrix", "vector_score")\
            .dialect(2)
    async def search(self, query_vec, k = 5):
        query_vec = query_vec.astype(np.float32)
        
        query = self.vector_query(            
                number_of_results = k
        )        
        result_vecs = []
        for vec in query_vec:
            # print(f"vec shape: {vec.shape}")
            query_params = { "vec_param": vec.tobytes()}
            result = await self.redis_conn.ft(self.index_name).search(
                query,
                query_params = query_params
            )
            result_vecs.append(result)
        return result_vecs
    
    async def fetch_one(self, key, field=None):
        print(f"key: {key}")
        if field:
            return await self.redis_conn.hget(key, field)
        else:
            return await self.redis_conn.hgetall(key)
    
class Retriver:
    def __init__(self, prefix):
        self.prefix = prefix
    def get_key(self, key):
        return f"{self.prefix}:{key}"  
    def search(self):
        pass
    
class QuerySource:
    def __init__(self):
        self.queries=[
            "How to use kernels in data science?"
        ]
    def read(self):
        return self.queries

import io
import matplotlib.pyplot as plt
import base64

def get_interaction_image(scores, doc_tokens, query_tokens):
    (x_shape, y_shape) = scores.shape
    
    fig = plt.figure(figsize=(len(doc_tokens),len(query_tokens)))
    plt.matshow(
      scores
    )
    
    plt.xlabel('doc vecs')
    plt.ylabel('query vecs')
    plt.xticks(range(y_shape), doc_tokens, fontsize=8)
    plt.xticks(rotation=90)
    plt.yticks(range(x_shape), query_tokens, fontsize=8)
    
    plt.tick_params(axis='x', which='major', labelsize=8)
    plt.tick_params(axis='y', which='major', labelsize=8)
    plt.tight_layout()
    
    imageIObytes = io.BytesIO()
    plt.savefig(imageIObytes, format='jpg', bbox_inches='tight')
    imageIObytes.seek(0)
    plt_base64_jpgData = base64.b64encode(imageIObytes.read())
    return plt_base64_jpgData


async def main():
    query_source = QuerySource()
    
    indexer = RedisIndexer(index_name="doc")
    await indexer.init_redis()

    model = ColBERTModel()
    
    aggregator = []
    for query in query_source.read():
        embedding = model.compute_query_representation(query)        
        results = await indexer.search(embedding, 100)
        aggregator.extend(results)
    # print(aggregator)
    
    retriver = Retriver(prefix="doc")
        
    uniq_docs={}
    for result in aggregator:
        # print(result)
        for doc in result.docs:
            if doc.doc_id not in uniq_docs.keys():
                uniq_docs[doc.doc_id] = {
                    "doc_id": doc.doc_id,
                    "doc": doc.doc,
                    "vector_matrix": doc.vector_matrix
                }
    
    
    # docs = [await indexer.fetch_one(retriver.get_key(f"{doc_id}-1"), "doc")
    #         for doc_id in doc_ids]
    # print(f"doc_ids: {doc_ids}")
    # print(f"doc_ids: {docs}")
    print(list(uniq_docs.values())[0]["doc"])
    # reranker = Reranker()
    late_interaction_ranking = []
    for each in uniq_docs.values():
        score = model.compute_score(query, each["doc"])
        each["late_interaction_score"] = score
        late_interaction_ranking.append(each)
#         reranker.write(score, doc)
#     reranker.rank()
    late_interaction_ranking.sort(
        key=lambda x: x["late_interaction_score"]
    )

#     reranker.render()
    print(late_interaction_ranking)

    print([
        (each["doc_id"], each["late_interaction_score"] , each["doc"])
        for each in late_interaction_ranking
    ])    
    
    # source = Source(Doc, Query)
    # sink =[]
    # for doc,query in source:
    #     interaction_map = model.compute_interaction(score,doc)
    #     query_tokens = get_tokens(query)
    #     doc_tokens = get_tokens(doc)
    #     rendered_image = render_image(
    #         interaction_map, 
    #         query_tokens, 
    #         doc_tokens
    #     )
    #     sink.write(rendered_image)
    # sink.serve()

    for each in late_interaction_ranking:
        score_map, doc_tokens, query_tokens = model.compute_interaction_map(
            query, each["doc"]
        )
        print(score_map.shape)
        interaction_map = get_interaction_image(score_map, doc_tokens, query_tokens)
        each["interaction_map"]=interaction_map
    print(len(late_interaction_ranking))
    # print(late_interaction_ranking[0]["interaction_map"])
    
        
    
if __name__ == '__main__':
    asyncio.run(
        main()
    )
    
    
    
    