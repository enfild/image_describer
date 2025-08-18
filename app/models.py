from pydantic import BaseModel
from typing import List, Dict, Any

class CaptionRequest(BaseModel):
    image_path: str
    model: str

class CaptionResponse(BaseModel):
    image_path: str
    model: str
    captions: List[str]

class HistoryResponse(BaseModel):
    entries: List[Dict[str, Any]]
