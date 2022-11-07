import asyncio
import typing as t
import redis.asyncio as redis

from fastapi import APIRouter
from vecsim_app import config
from vecsim_app.embeddings import Embeddings
from vecsim_app.models import Paper

from vecsim_app.schema import (
    SimilarityRequest,
    UserTextSimilarityRequest
)
from vecsim_app.search_index import SearchIndex

from vecsim_app.retrival import ColBERTModel,RedisIndexer
from vecsim_app.retrival import gather_with_concurrency, get_interaction_image

paper_router = r = APIRouter()
redis_client = redis.from_url(config.REDIS_URL)
embeddings = Embeddings()
search_index = SearchIndex()

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
    limit: int = 20,
    skip: int = 0,
    years: str = "",
    categories: str = ""
):
    papers = []
    expressions = []
    years = [year for year in years.split(",") if year]
    categories = [cat for cat in categories.split(",") if cat]
    if years and categories:
        expressions.append(
            (Paper.year << years) & \
            (Paper.categories << categories)
        )
    elif years and not categories:
        expressions.append(Paper.year << years)
    elif categories and not years:
        expressions.append(Paper.categories << categories)
    # Run query

    papers = await Paper.find(*expressions)\
        .copy(offset=skip, limit=limit)\
        .execute(exhaust_results=False)

    # Get total count
    total = (
        await redis_client.ft(config.INDEX_NAME).search(
            search_index.count_query(years=years, categories=categories)
        )
    ).total
    return {
        'total': total,
        'papers': papers
    }


@r.post("/vectorsearch/text", response_model=t.Dict)
async def find_papers_by_text(similarity_request: SimilarityRequest):
    # Create query
    query = search_index.vector_query(
        similarity_request.categories,
        similarity_request.years,
        similarity_request.search_type,
        similarity_request.number_of_results
    )
    count_query = search_index.count_query(
        years=similarity_request.years,
        categories=similarity_request.categories
    )

    # find the vector of the Paper listed in the request
    paper_vector_key = "paper_vector:" + str(similarity_request.paper_id)
    vector = await redis_client.hget(paper_vector_key, "vector")

    # obtain results of the queries
    total, results = await asyncio.gather(
        redis_client.ft(config.INDEX_NAME).search(count_query),
        redis_client.ft(config.INDEX_NAME).search(query, query_params={"vec_param": vector})
    )

    # Get Paper records of those results
    return await papers_from_results(total.total, results)


@r.post("/vectorsearch/text/user", response_model=t.Dict)
async def find_papers_by_user_text(similarity_request: UserTextSimilarityRequest):
    limit=3
    run_cpu =True

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
    if run_cpu :
        uniq_docs= uniq_docs[:limit]
    ## Ranking
    late_interaction_ranking = []
    for each in uniq_docs:    
        doc = str(each["doc"])
        score = model.compute_score(query, doc)
        each["late_interaction_score"] = score
        late_interaction_ranking.append(each)
    #         reranker.write(score, doc)
    late_interaction_ranking.sort(
        key=lambda x: x["late_interaction_score"]
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
    total = limit #len(uniq_doc_ids)
    results = late_interaction_ranking

    # Get Paper records of those results
    return await papers_from_results(total, late_interaction_ranking)
