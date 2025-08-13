# 🧠 Asistente Virtual con Memoria (RAG + FastAPI)

Backend que implementa un asistente en español con memoria por conversación y búsqueda semántica sobre PDFs (RAG).
Stack principal: FastAPI, FAISS, Sentence-Transformers, Ollama (Mistral) y PostgreSQL.

---

## ✅ Requisitos técnicos

### 📦 `requirements.txt`
```txt
fastapi
uvicorn
sqlalchemy
psycopg2-binary
python-jose[cryptography]
passlib[bcrypt]
sentence-transformers
pymupdf
faiss-cpu
requests
email-validator
python-dotenv
```

## 📁 Estructura del Proyecto

```
backend-mawell/
├── app/
│   ├── api/                    # Endpoints (chat, debug)
│   ├── models/                 # SQLAlchemy models
│   ├── schemas/                # Pydantic schemas
│   ├── services/               # IA (RAG), embeddings, auth
│   ├── config.py               # DB & app config
│   └── main.py                 # FastAPI app
├── data/
│   ├── pdfs/                   # PDFs a indexar
│   └── vector_db/              # FAISS (index.faiss) + docs.pkl
├── scripts/
│   └── create_index.py         # Script de indexación masiva
├── media/
├── requirements.txt
├── .env.example
└── README.md
```

### ⚙️ Entorno virtual
```bash
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

---

## 🧠 IA y vectorización (RAG)

### 🦙 Ollama
- Instalar desde: https://ollama.com
- Iniciar modelo:
```bash
ollama run mistral
```

---

## 🛠️ Comandos útiles
Ejecutar script para cargar y vectorizar PDF

- Cargar y trocear texto de PDFs sube archivos pdfs en /data/pdfs y ejecuta
```bash
python -m scripts.create_index
```

- Iniciar servidor FastAPI
```bash
uvicorn app.main:app --reload --port 8000
```