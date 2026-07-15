"""
Interfaz web (Gradio) para el sistema RAG sobre arXiv Paper Abstracts.

Pensada para desplegarse en Hugging Face Spaces (SDK Gradio). Cada consulta
se procesa de forma independiente, sin memoria conversacional.
"""

import os

import gradio as gr

from src.rag import rag_pipeline

# En Hugging Face Spaces, GEMINI_API_KEY se configura como Secret y
# queda disponible como variable de entorno del proceso.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


def _formatear_evidencias(docs: list[dict]) -> str:
    """Arma un bloque de Markdown con las evidencias recuperadas."""
    if not docs:
        return "_No se recuperaron documentos para esta consulta._"

    bloques = []
    for i, doc in enumerate(docs, start=1):
        titulo = doc.get("title", "(sin título)")
        terms = doc.get("terms", "")
        distancia = doc.get("distance", 0.0)
        abstract = doc.get("abstract", "")
        bloques.append(
            f"**[Doc {i}] {titulo}**\n\n"
            f"Categorías: `{terms}`  |  Distancia: `{distancia:.4f}`\n\n"
            f"<details><summary>Ver abstract</summary>\n\n{abstract}\n\n</details>"
        )
    return "\n\n---\n\n".join(bloques)


def responder(mensaje: str, historial: list):
    """Callback del chat: recupera evidencias y genera la respuesta.

    El historial usa el formato "messages" de gr.Chatbot (lista de dicts
    con claves "role" y "content").
    """
    if not mensaje or not mensaje.strip():
        historial = historial + [
            {"role": "user", "content": mensaje},
            {"role": "assistant", "content": "Por favor escribe una consulta."},
        ]
        return historial, ""

    respuesta, docs = rag_pipeline(mensaje)
    evidencias_md = _formatear_evidencias(docs)

    historial = historial + [
        {"role": "user", "content": mensaje},
        {"role": "assistant", "content": respuesta},
    ]
    return historial, evidencias_md


with gr.Blocks(title="RAG arXiv Paper Abstracts") as demo:
    gr.Markdown(
        "# Sistema RAG sobre arXiv Paper Abstracts\n"
        "Realiza consultas en lenguaje natural sobre un corpus de resúmenes "
        "de artículos científicos. El sistema recupera los documentos más "
        "relevantes mediante búsqueda semántica y genera una respuesta con "
        "un LLM (Gemini), citando las evidencias utilizadas.\n\n"
        "Cada consulta se procesa de forma independiente (sin memoria "
        "conversacional)."
    )

    if not GEMINI_API_KEY:
        gr.Markdown(
            "**Aviso:** no se detectó la variable de entorno `GEMINI_API_KEY`. "
            "La recuperación de documentos funcionará, pero la generación de "
            "respuestas mostrará un mensaje de error hasta que se configure la clave."
        )

    chatbot = gr.Chatbot(label="Conversación", height=400)

    with gr.Row():
        entrada = gr.Textbox(
            label="Consulta",
            placeholder="Ej: What are the main applications of Graph Neural Networks?",
            scale=4,
        )
        boton_enviar = gr.Button("Enviar", scale=1)

    gr.Markdown("## Evidencias recuperadas")
    panel_evidencias = gr.Markdown(
        "_Las evidencias de la última consulta aparecerán aquí._"
    )

    def _enviar(mensaje, historial):
        nuevo_historial, evidencias = responder(mensaje, historial)
        return nuevo_historial, evidencias, ""

    boton_enviar.click(
        _enviar,
        inputs=[entrada, chatbot],
        outputs=[chatbot, panel_evidencias, entrada],
    )
    entrada.submit(
        _enviar,
        inputs=[entrada, chatbot],
        outputs=[chatbot, panel_evidencias, entrada],
    )

if __name__ == "__main__":
    demo.launch()
