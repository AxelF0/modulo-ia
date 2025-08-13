# servidor/modulo-ia/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import Base, engine
from app.api import chat_router, debug_router
from app.api.whatsapp_integration import router as whatsapp_router
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Módulo IA - Sistema WhatsApp Inmobiliario",
    description="IA con RAG para asistente inmobiliario integrado con WhatsApp",
    version="2.0.0",
)

# CORS para permitir peticiones del sistema WhatsApp
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Gateway
        "http://localhost:3002",  # Procesamiento
        "http://localhost:3005",  # Respuestas
        "*"  # En desarrollo
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crear tablas si no existen
Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {
        "message": "Módulo IA WhatsApp activo 🤖",
        "service": "ia-module",
        "version": "2.0.0",
        "endpoints": {
            "whatsapp": "/api/ia/process-query",
            "chat": "/chat/send",
            "debug": "/debug/search",
            "health": "/api/ia/health"
        }
    }

@app.get("/api/health")
def health_check():
    """Health check para el gateway"""
    return {
        "success": True,
        "service": "ia-module",
        "status": "healthy",
        "port": int(os.getenv("IA_PORT", "3003"))
    }

# Incluir routers
app.include_router(whatsapp_router)  # NUEVO: Integración WhatsApp
app.include_router(chat_router)      # Chat original
app.include_router(debug_router)     # Debug

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("IA_PORT", "3003"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)