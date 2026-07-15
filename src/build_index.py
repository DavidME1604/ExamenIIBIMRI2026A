"""
Construcción del índice vectorial con ChromaDB.

Lee el corpus limpio (data/corpus_clean.csv), genera embeddings con
sentence-transformers (all-MiniLM-L6-v2) a partir de "título + abstract"
y los guarda en una colección persistente de ChromaDB (chroma_db/).
"""

import os
import sys

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION_NAME,
    CORPUS_CSV_PATH,
    EMBEDDING_MODEL,
)

BATCH_SIZE_EMBEDDINGS = 512  # lote para calcular embeddings
BATCH_SIZE_CHROMA = 5000  # límite de elementos por llamada a collection.add


def construir_indice():
    if not os.path.exists(CORPUS_CSV_PATH):
        raise FileNotFoundError(
            f"No existe {CORPUS_CSV_PATH}. Ejecuta primero src/data_prep.py."
        )

    print(f"Leyendo corpus desde: {CORPUS_CSV_PATH}")
    df = pd.read_csv(CORPUS_CSV_PATH)
    df = df.dropna(subset=["titles", "summaries"]).reset_index(drop=True)
    print(f"Documentos a indexar: {len(df)}")

    # Texto a embeber: título + abstract concatenados.
    textos = (df["titles"] + ". " + df["summaries"]).tolist()

    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL}")
    modelo = SentenceTransformer(EMBEDDING_MODEL)

    print("Calculando embeddings...")
    embeddings = modelo.encode(
        textos,
        batch_size=BATCH_SIZE_EMBEDDINGS,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Cliente persistente de ChromaDB.
    cliente = chromadb.PersistentClient(path=CHROMA_DIR)

    # Si la colección ya existe (de una ejecución previa), la recreamos
    # para evitar duplicados o inconsistencias.
    try:
        cliente.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # embedding_function=None: nosotros calculamos y pasamos los embeddings,
    # Chroma no debe usar su función de embeddings por defecto.
    coleccion = cliente.create_collection(
        name=COLLECTION_NAME, embedding_function=None
    )

    ids = [str(i) for i in range(len(df))]
    metadatas = [
        {"title": row["titles"], "terms": row["terms"]}
        for _, row in df.iterrows()
    ]
    documentos = df["summaries"].tolist()

    print("Guardando en ChromaDB por lotes...")
    total = len(df)
    for inicio in range(0, total, BATCH_SIZE_CHROMA):
        fin = min(inicio + BATCH_SIZE_CHROMA, total)
        coleccion.add(
            ids=ids[inicio:fin],
            embeddings=embeddings[inicio:fin].tolist(),
            metadatas=metadatas[inicio:fin],
            documents=documentos[inicio:fin],
        )
        print(f"  Lote guardado: {inicio} - {fin}")

    print(f"Índice construido con {coleccion.count()} documentos en {CHROMA_DIR}")


if __name__ == "__main__":
    construir_indice()
