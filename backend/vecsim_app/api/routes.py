import asyncio
import typing as t
import redis.asyncio as redis

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from vecsim_app import config

from vecsim_app.schema import  UserTextSimilarityRequest, UserTextSimilarityExplainRequest

from vecsim_app.retrival import ColBERTModel,RedisIndexer
from vecsim_app.retrival import gather_with_concurrency, get_interaction_image

paper_router = r = APIRouter()
redis_client = redis.from_url(config.REDIS_URL)

async def process_paper(p, i: int) -> t.Dict[str, t.Any]:    
    paper = p
    score = float(p['late_interaction_score'])
    paper['similarity_score'] = score
    return paper

async def papers_from_results(total, results) -> t.Dict[str, t.Any]:
    # extract papers from VSS results
    return {
        'total': total,
        'papers': [
            await process_paper(p, i)
            for i, p in enumerate(results)
        ]
    }

@r.get("/", response_model=t.Dict)
async def get_papers(
    paper_id: str = ""
):
    indexer = RedisIndexer(index_name="doc_vec", url=config.REDIS_URL)
    await indexer.init_redis()
    
    uniq_doc_ids =[paper_id]
    uniq_docs = await gather_with_concurrency(indexer.redis_conn, *uniq_doc_ids, field='doc')
    uniq_docs =[ doc for doc in uniq_docs if doc["doc"] is not None]

    ## Ranking
    late_interaction_ranking = []
    for each in uniq_docs:    
        doc = str(each["doc"])
        each["late_interaction_score"] = 0
        late_interaction_ranking.append(each)
    #         reranker.write(score, doc)
    late_interaction_ranking.sort(
        key=lambda x: x["late_interaction_score"],
        reverse=True
    )

    # obtain results of the queries
    total = len(late_interaction_ranking)
    results = late_interaction_ranking

    # Get Paper records of those results
    return await papers_from_results(total, late_interaction_ranking)


@r.post("/vectorsearch/text", response_model=t.Dict)
async def find_papers_by_user_text(similarity_request: UserTextSimilarityRequest):
    limit=similarity_request.number_of_results
    
    indexer = RedisIndexer(index_name="doc_vec", url=config.REDIS_URL)
    await indexer.init_redis()
    model = ColBERTModel()

    ## search
    aggregator = []
    for query in [similarity_request.user_text]:
        embedding = model.compute_query_representation(query)        
        results = await indexer.search(embedding, 100)
        aggregator.extend(results)
    # print(aggregator)
    ## aggregate doc_id
    uniq_doc_ids=[]
    for result in aggregator:
        # print(result)
        for doc in result.docs:
            uniq_doc_ids.append(doc.doc_id)
    uniq_doc_ids = list(set(uniq_doc_ids))
    # print("uniq_doc_ids:", uniq_doc_ids)
    # print("uniq_doc_ids len:", len(uniq_doc_ids))
 
    ## Retrival
    uniq_docs = await gather_with_concurrency(indexer.redis_conn, *uniq_doc_ids, field='doc')
    uniq_docs =[ doc for doc in uniq_docs if doc["doc"] is not None]
    
    ## Ranking
    late_interaction_ranking = []
    for each in uniq_docs:    
        doc = str(each["doc"])
        score = model.compute_score(query, doc)
        each["late_interaction_score"] = score
        late_interaction_ranking.append(each)
    #         reranker.write(score, doc)
    late_interaction_ranking.sort(
        key=lambda x: x["late_interaction_score"],
        reverse=True
    )

    # limit entries returned back
    late_interaction_ranking= late_interaction_ranking[:limit]

    ## explainability
    for each in late_interaction_ranking:
        score_map, doc_tokens, query_tokens = model.compute_interaction_map(
            query, str(each["doc"])
        )
        # print(score_map.shape)
        interaction_map = get_interaction_image(score_map, doc_tokens, query_tokens)
        each["interaction_map"]=interaction_map
    # print(len(late_interaction_ranking))
    # print(late_interaction_ranking[0]["interaction_map"])


    # obtain results of the queries
    total = len(late_interaction_ranking)
    results = late_interaction_ranking

    # Get Paper records of those results
    return await papers_from_results(total, late_interaction_ranking)


@r.post("/vectorsearch/text/explain", response_model=t.Dict)
async def find_papers_by_user_text(similarity_request: UserTextSimilarityExplainRequest):
    indexer = RedisIndexer(index_name="doc_vec", url=config.REDIS_URL)
    await indexer.init_redis()

    ## search
    model = ColBERTModel()
    aggregator = []
    for query in [similarity_request.user_text]:
        embedding = model.compute_query_representation(query)        
        results = await indexer.search(embedding, 100)
        aggregator.extend(results)
    # print(aggregator)
    
    uniq_doc_ids=[similarity_request.paper_id]    
    ## Retrival
    uniq_docs = await gather_with_concurrency(indexer.redis_conn, *uniq_doc_ids, field='doc')
    uniq_docs =[ doc for doc in uniq_docs if doc["doc"] is not None]

    ## Ranking
    late_interaction_ranking = []
    for each in uniq_docs:    
        doc = str(each["doc"])
        score = model.compute_score(query, doc)
        each["late_interaction_score"] = score
        late_interaction_ranking.append(each)
    #         reranker.write(score, doc)
    late_interaction_ranking.sort(
        key=lambda x: x["late_interaction_score"],
        reverse=True
    )

    ## explainability
    for each in late_interaction_ranking:
        score_map, doc_tokens, query_tokens = model.compute_interaction_map(
            query, str(each["doc"])
        )
        # print(score_map.shape)
        interaction_map = get_interaction_image(score_map, doc_tokens, query_tokens)
        each["interaction_map"]=interaction_map
    # print(len(late_interaction_ranking))
    # print(late_interaction_ranking[0]["interaction_map"])


    # obtain results of the queries
    total = len(late_interaction_ranking)
    results = late_interaction_ranking

    # Get Paper records of those results
    # return await papers_from_results(total, late_interaction_ranking)
    if len(late_interaction_ranking ) >0:    
        return StreamingResponse(late_interaction_ranking[0]["interaction_map"], media_type="image/png")
    else:
        return {
            "message":"Something went wrong while computing interaction map"
        }

