from pydantic import BaseModel
from datetime import datetime

class MessageResponse(BaseModel):
    id: int
    question: str
    answer: str
    timestamp: datetime

    class Config:
        from_attributes = True
