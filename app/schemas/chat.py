from pydantic import BaseModel

class ChatRequest(BaseModel):
    conversation_id: int
    question: str
