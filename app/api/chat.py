# app/api/chat.py
# ---------------------------------------------------------------------
# Chat API:
# - Small talk intent (greetings) answered without RAG for better UX
# - Knowledge queries go through RAG; if no context over threshold,
#   guide the user toward topics actually present in the PDFs
# - Persists conversation messages in DB
# ---------------------------------------------------------------------

import os
import re
import requests
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session

from app.models import Conversation, Message
from app.schemas.chat import ChatRequest
from app.schemas.message import MessageResponse
from app.schemas.conversation import ConversationSummary, ConversationCreate, ConversationResponse
from app.config import SessionLocal

from app.services.ia_service import (
    ask_mistral_with_context,
    summarize_pdf,
    build_softgrounded_reply,
    get_suggested_titles,
    format_topics_inline,
)
HELP_PAT = re.compile(r"(con\s*qué|en\s*qué)\s+(me\s+)?puedes\s+ayudar|qué\s+puedes\s+hacer|capacidades|ayuda", re.I)
PDF_OVERVIEW_PAT = re.compile(r"(qué|que)\s+informaci[oó]n\s+.*(hay|encontrar[eé])\s+en\s+el\s+documento\s+de\s+(.+?\.pdf)", re.I)
router = APIRouter(prefix="/chat", tags=["chat"])

# --- DB dependency ----------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Intent routing patterns -----------------------------------------
GREETING_PAT = re.compile(
    r"^\s*(hola|holi|hey|qué tal|que tal|buenas|buenos días|buenas tardes|buenas noches)\b",
    re.I
)

# --- Plain LLM call for small talk (no RAG) --------------------------
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "mistral")

def llm_freeform(prompt: str) -> str:
    """
    Plain LLM call without RAG for small talk / UX niceties.
    Keep it short and helpful.
    """
    sys = "Eres un asistente en español, cordial, breve y claro."
    final = f"{sys}\n\nInstrucción del usuario: {prompt}\nRespuesta:"
    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL_NAME, "prompt": final, "stream": False},
            timeout=60,
        )
        if resp.status_code != 200:
            return "¡Hola! ¿En qué puedo ayudarte hoy?"
        return resp.json().get("response", "").strip() or "¡Hola! ¿En qué puedo ayudarte hoy?"
    except requests.RequestException:
        return "¡Hola! ¿En qué puedo ayudarte hoy?"

# --- Routes -----------------------------------------------------------

@router.post("/start", response_model=ConversationResponse)
def start_conversation(data: ConversationCreate, db: Session = Depends(get_db)):
    """
    Create a new conversation entry.
    """
    new_convo = Conversation(title=data.title)
    db.add(new_convo)
    db.commit()
    db.refresh(new_convo)
    return new_convo

@router.post("/send", response_model=MessageResponse)
def send_question(data: ChatRequest, db: Session = Depends(get_db)):
    """
    Handle user message:
    - Build lightweight history
    - If greeting: small talk without RAG
    - Else: Knowledge flow; if no relevant context → natural guidance
    - Persist message to DB
    """
    # 1) Load history
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == data.conversation_id)
        .order_by(Message.timestamp)
        .all()
    )

    # 2) Build lightweight history (user/assistant pairs)
    history_lines = [f"Usuario: {m.question}\nAsistente: {m.answer}" for m in messages]
    history_lines.append(f"Usuario: {data.question}")
    history = "\n".join(history_lines)

    user_text = (data.question or "").strip()

    # 3) Intent routing
    if GREETING_PAT.search(user_text):
        # Small talk + sugerencias suaves (opcionales)
        titles = get_suggested_titles(user_text, max_suggestions=4)
        topics_line = format_topics_inline(titles, max_items=4)
        prompt = (
            "Saluda cordialmente en una sola frase y ofrece ayuda de manera breve. "
            "No menciones detalles técnicos. "
            # + (f"Si encaja natural, sugiere continuar con: {topics_line}. " if topics_line else "")
            # + "Termina con una pregunta corta para invitar a elegir un tema."
        )
        answer = llm_freeform(prompt)

    elif HELP_PAT.search(user_text):
        # Freeform con capacidades y sugerencias, sin internals ni métricas
        titles = get_suggested_titles(user_text, max_suggestions=5)
        topics_line = format_topics_inline(titles, max_items=5)
        prompt = (
            "Responde en tono cordial y claro, sin tecnicismos. "
            "Di en una frase que puedes ayudar con explicaciones, resúmenes, definiciones y comparaciones. "
            + (f"Luego sugiere continuar por temas como: {topics_line}. " if topics_line else "")
            + "Cierra con: ‘¿Sobre cuál te gustaría saber más?’"
        )
        answer = llm_freeform(prompt)

    else:
        # Knowledge path → RAG
        m = PDF_OVERVIEW_PAT.search(user_text)  # “¿qué hay en X.pdf?”
        if m:
            pdf_name = m.group(3).strip()
            info = summarize_pdf(pdf_name)
            titles = info.get("titles", [])
            if not titles:
                answer = (
                    f"No tengo información disponible sobre “{pdf_name}” por ahora. "
                    "Indícame otro tema o sección y lo revisamos."
                )
            else:
                bullets = "\n".join(f"• {t['title']}" for t in titles[:8] if t.get("title"))
                answer = (
                    f"En “{pdf_name}” encontrarás temas como:\n{bullets}\n\n"
                    "Si te interesa alguno, dime y lo exploramos."
                )
        else:
            # RAG normal
            result = ask_mistral_with_context(query=user_text, history=history)
            if result.get("used_context"):
                answer = result.get("answer", "No se pudo generar respuesta.")
            else:
                # Sin contexto suficiente → también freeform con sugerencias (natural)
                titles = get_suggested_titles(user_text, max_suggestions=5)
                topics_line = format_topics_inline(titles, max_items=5)
                prompt = (
                    "Responde en tono natural y breve, sin explicar tu funcionamiento. "
                    "Invita a seguir con temas relacionados de los documentos. "
                    + (f"Propón continuar con: {topics_line}. " if topics_line else "")
                    + "Termina con una pregunta corta para que el usuario elija un tema."
                )
                answer = llm_freeform(prompt)

    # 4) Persist message
    new_msg = Message(
        conversation_id=data.conversation_id,
        question=user_text,
        answer=answer
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)

    return new_msg

@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def get_conversation_messages(conversation_id: int, db: Session = Depends(get_db)):
    """
    Return all messages in a conversation sorted by timestamp.
    """
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp)
        .all()
    )
    return messages

@router.get("/my-conversations", response_model=list[ConversationSummary])
def list_conversations(db: Session = Depends(get_db)):
    """
    List conversations ordered by creation time (desc).
    """
    conversations = db.query(Conversation).order_by(Conversation.created_at.desc()).all()
    return conversations

@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """
    Delete conversation by id (cascade deletes messages).
    """
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    db.delete(conversation)
    db.commit()
