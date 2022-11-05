from indexer import Indexer
from retrival import SearchIndex # Retriver
import time
import pandas as pd
import pickle

import asyncio

from redis.commands.search.field import (
    TagField,
    VectorField,
    NumericField,
    TextField
)


# Load papers dataframe
def read_paper_df(pickled_data_df_file) -> pd.DataFrame:
    with open(pickled_data_df_file, "rb") as f:
        df = pickle.load(f)
    return df

async def main(data_file):
    df = read_paper_df(data_file)
    print(df.head())
    
    # preprocessing 
    df['input'] = df.apply(lambda r: r.title + r.abstract, axis=1)
    df.reset_index(drop=True, inplace=True)
    
    # for farster dev 
    # TODO: remove
    df = df.sample(frac=0.1)
    
    print('Indexing..')
    indexer = Indexer()
    
    await indexer.index(df.input.to_list()) #[:5]
    
    print()
    print('Indexing completed')
    
    await indexer.print_db_size()
    
    print('Search')
    # doc = df.input.to_list()[0]
    # doc_vector = indexer.get_vectors(doc)
        
    # create search index 
    
    search_index = SearchIndex('docs', indexer.redis_conn)    
    await search_index.delete()
    
    print('creating redis index')
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
    await search_index.create(
        vector_field,
        prefix="docs:"
    )
    # -----------------

    print('redis search')    
    result = await search(
        indexer.redis_conn, 
        search_index, 
        10,
        indexer.get_key("1-1")
    )
    
    print(result)
    
    

    
async def search(redis_conn, search_index, k: int, doc_id: str):
    query = search_index.vector_query(            
            number_of_results = k
    )
    
    print("searching for nearest neighbors to", doc_id)
    start = time.time()
    vector = await redis_conn.hget(doc_id, "vector")
    query_params={"vec_param": vector}
    print(query_params)
    
    result = await redis_conn.ft(search_index.index_name).search(
        query,
        query_params=query_params
    )
    print("done in", time.time()-start, "seconds")
    return result
    
    
    
    



if __name__ == '__main__':
    DATA_FILE = "../arxiv_papers_df.pkl"
    asyncio.run(
        main(DATA_FILE)
    )