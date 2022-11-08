from pydantic import BaseModel

class UserTextSimilarityRequest(BaseModel):
    user_text: str
    number_of_results: int = 10
    search_type: str = "KNN"
    interaction_map: bool = False

class UserTextSimilarityExplainRequest(BaseModel):
    user_text: str
    paper_id: str    
    interaction_map: bool = True