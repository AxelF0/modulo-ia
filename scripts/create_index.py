# /scripts/create_index.py

import os
from app.services.embedding_service import build_vector_index

# Carpeta donde están los PDFs
pdf_directory = os.getenv("PDF_SOURCE_PATH", "data/pdfs")

# Función para procesar todos los PDFs en la carpeta
def process_pdfs_in_directory(pdf_directory):
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):  # Solo procesamos los archivos .pdf
            pdf_path = os.path.join(pdf_directory, filename)
            print(f"Procesando el archivo: {pdf_path}")
            build_vector_index(pdf_path, max_chars=1000, overlap=180)  # Llamamos a la función que procesa cada PDF

# Llamamos a la función para procesar todos los PDFs en la carpeta
process_pdfs_in_directory(pdf_directory)
