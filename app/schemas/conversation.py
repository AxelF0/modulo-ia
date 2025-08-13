from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ConversationCreate(BaseModel):
    title: Optional[str] = "Nueva conversación"

class ConversationResponse(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationSummary(BaseModel):
    id: int
    title: str
    created_at: datetime

    class Config:
        from_attributes = True