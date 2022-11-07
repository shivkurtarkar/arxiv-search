import asyncio
import typing as t
import redis.asyncio as redis

from fastapi import APIRouter
from vecsim_app import config
# from vecsim_app.embeddings import Embeddings
from vecsim_app.models import Paper

from vecsim_app.schema import (
    SimilarityRequest,
    UserTextSimilarityRequest
)

# from vecsim_app.search_index import SearchIndex

paper_router = r = APIRouter()
redis_client = redis.from_url(config.REDIS_URL)
# embeddings = Embeddings()
# search_index = SearchIndex()

async def process_paper(p, i: int)-> t.Dict[str, t.Any]:
    """ TODO:
    """
    # paper = await Paper.get(p.paper_pk)
    paper = dict()
    score = 1 
    paper['similarity_score']=score
    return paper

async def papers_from_results(totol, results) -> t.Dict[str, t.Any]:
    # extract papers from VSS results
    return {
        'total': total,
        'papers': [
            await process_paper(p, i)
            for i, p in enumerate(results.docs)
        ]
    }

@r.get("/", response_model=t.Dict)
async def get_papers(
    limit:int =20,
    skip: int = 0,
    years: str = "",
    categories: str = ""
):

    """ TODO:
    """
    total =0
    papers=[]

    return {
        'total': total,
        'papers':papers
    }

@r.post("/vectorsearch/text", response_model=t.Dict)
async def find_papers_by_text(similarity_request: SimilarityRequest):
    """
    """
    total=0
    results=[]
    return papers_from_results(total, results)

@r.post("/vectrosearch/text/user", response_model=t.Dict)
async def find_papers_by_user_text(similarity_request: UserTextSimilarityRequest):
    total=0
    results=[]
    return papers_from_results(total, results)