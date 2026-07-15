"""
Núcleo del sistema RAG: recuperación semántica + generación con Gemini.

- retrieve(query, top_k): busca los documentos más relevantes en ChromaDB.
- generate_answer(query, docs): genera una respuesta con Gemini usando
  los documentos recuperados como contexto.
- rag_pipeline(query): junta ambos pasos y devuelve (respuesta, docs).
"""

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    TOP_K,
)

# Cargamos el modelo de embeddings y el cliente de Chroma una sola vez,
# para no recargarlos en cada consulta.
_modelo_embeddings = None
_coleccion = None


def _obtener_modelo_embeddings():
    global _modelo_embeddings
    if _modelo_embeddings is None:
        _modelo_embeddings = SentenceTransformer(EMBEDDING_MODEL)
    return _modelo_embeddings


def _obtener_coleccion():
    global _coleccion
    if _coleccion is None:
        cliente = chromadb.PersistentClient(path=CHROMA_DIR)
        _coleccion = cliente.get_collection(COLLECTION_NAME)
    return _coleccion


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Recupera los `top_k` documentos más similares a la consulta.

    Devuelve una lista de diccionarios con las claves:
    title, abstract, terms, distance.
    """
    modelo = _obtener_modelo_embeddings()
    coleccion = _obtener_coleccion()

    embedding_consulta = modelo.encode([query], convert_to_numpy=True).tolist()

    resultados = coleccion.query(
        query_embeddings=embedding_consulta,
        n_results=top_k,
    )

    documentos = []
    metadatas = resultados.get("metadatas", [[]])[0]
    textos = resultados.get("documents", [[]])[0]
    distancias = resultados.get("distances", [[]])[0]

    for metadato, texto, distancia in zip(metadatas, textos, distancias):
        documentos.append(
            {
                "title": metadato.get("title", ""),
                "abstract": texto,
                "terms": metadato.get("terms", ""),
                "distance": distancia,
            }
        )
    return documentos


def _construir_prompt(query: str, docs: list[dict]) -> str:
    """Arma el prompt con instrucciones estrictas de fidelidad al contexto."""
    contexto = "\n\n".join(
        f"[Doc {i + 1}] Título: {doc['title']}\n"
        f"Categorías: {doc['terms']}\n"
        f"Abstract: {doc['abstract']}"
        for i, doc in enumerate(docs)
    )

    prompt = f"""Eres un asistente de investigación que responde preguntas basándote
ÚNICAMENTE en los abstracts de artículos científicos proporcionados como contexto.

Reglas estrictas:
1. Responde SOLO usando la información contenida en los documentos de contexto.
2. Cuando uses información de un documento, cita su número entre corchetes,
   por ejemplo [Doc 1], [Doc 2], etc.
3. Si el contexto no contiene información suficiente para responder la consulta,
   dilo explícitamente en el idioma de la consulta (por ejemplo, en español:
   "El corpus no contiene información suficiente para responder esta consulta.";
   en inglés: "The corpus does not contain enough information to answer this query.").
4. No inventes información que no esté en los documentos.
5. Responde en el mismo idioma en que fue formulada la consulta.

Contexto (documentos recuperados):
{contexto}

Consulta del usuario: {query}

Respuesta:"""
    return prompt


def generate_answer(query: str, docs: list[dict]) -> str:
    """
    Genera una respuesta con Gemini usando los documentos recuperados
    como contexto. Maneja errores de API devolviendo un mensaje claro.
    """
    from src.config import GEMINI_API_KEY

    if not GEMINI_API_KEY:
        return (
            "No se configuró GEMINI_API_KEY. Define esta variable de entorno "
            "(o el Secret en Hugging Face Spaces) para poder generar respuestas."
        )

    if not docs:
        return "El corpus no contiene información suficiente para responder esta consulta."

    prompt = _construir_prompt(query, docs)

    try:
        from google import genai

        cliente = genai.Client(api_key=GEMINI_API_KEY)
        respuesta = cliente.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return respuesta.text
    except Exception as error:
        return f"Ocurrió un error al generar la respuesta con Gemini: {error}"


def rag_pipeline(query: str, top_k: int = TOP_K):
    """Ejecuta el pipeline completo: recuperación + generación."""
    docs = retrieve(query, top_k=top_k)
    respuesta = generate_answer(query, docs)
    return respuesta, docs
