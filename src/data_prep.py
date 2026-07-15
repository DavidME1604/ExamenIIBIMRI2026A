"""
Preparación del corpus arXiv Paper Abstracts.

Descarga el dataset desde Kaggle (spsayakpaul/arxiv-paper-abstracts),
lo limpia y guarda un subconjunto en data/corpus_clean.csv listo para
ser indexado por src/build_index.py.

Columnas esperadas del CSV original: titles, abstracts, terms
(se renombra "abstracts" a "summaries" para uso interno del proyecto).
"""

import glob
import os
import re
import sys

import kagglehub
import pandas as pd

# Permite ejecutar este archivo tanto como script (python src/data_prep.py)
# como módulo importado (from src.data_prep import preparar_corpus).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CORPUS_CSV_PATH, DATA_DIR, MAX_DOCS  # noqa: E402


def _encontrar_csv(directorio: str) -> str:
    """Busca el archivo CSV del dataset dentro de la carpeta descargada."""
    candidatos = glob.glob(os.path.join(directorio, "**", "*.csv"), recursive=True)
    if not candidatos:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .csv en {directorio}"
        )
    # Si hay varios, tomamos el más grande (probablemente el dataset completo).
    candidatos.sort(key=os.path.getsize, reverse=True)
    return candidatos[0]


def _limpiar_texto(texto: str) -> str:
    """Quita saltos de línea y espacios repetidos de un texto."""
    if not isinstance(texto, str):
        return ""
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def preparar_corpus(max_docs: int = MAX_DOCS) -> pd.DataFrame:
    """
    Descarga el dataset con kagglehub, lo limpia y devuelve un DataFrame
    con como máximo `max_docs` filas.

    Pasos de limpieza:
    - Elimina duplicados por título.
    - Elimina filas con título o abstract nulos/vacíos.
    - Quita saltos de línea de los abstracts.
    - Se queda con las primeras `max_docs` filas.
    """
    print("Descargando dataset desde Kaggle (spsayakpaul/arxiv-paper-abstracts)...")
    ruta_dataset = kagglehub.dataset_download("spsayakpaul/arxiv-paper-abstracts")
    print(f"Dataset descargado en: {ruta_dataset}")

    ruta_csv = _encontrar_csv(ruta_dataset)
    print(f"Leyendo CSV: {ruta_csv}")
    df = pd.read_csv(ruta_csv)

    # El CSV original de Kaggle trae la columna "abstracts" (no "summaries").
    # La renombramos para tener un nombre interno consistente en el proyecto.
    if "abstracts" in df.columns and "summaries" not in df.columns:
        df = df.rename(columns={"abstracts": "summaries"})

    # Verificamos que existan las columnas esperadas.
    columnas_esperadas = {"titles", "summaries", "terms"}
    faltantes = columnas_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(
            f"El CSV no tiene las columnas esperadas {columnas_esperadas}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # Quitamos filas nulas en título o abstract.
    df = df.dropna(subset=["titles", "summaries"])

    # Limpiamos texto (quitamos saltos de línea, espacios extra).
    df["titles"] = df["titles"].astype(str).map(_limpiar_texto)
    df["summaries"] = df["summaries"].astype(str).map(_limpiar_texto)
    df["terms"] = df["terms"].astype(str).map(_limpiar_texto)

    # Quitamos filas que hayan quedado vacías tras la limpieza.
    df = df[(df["titles"] != "") & (df["summaries"] != "")]

    # Eliminamos duplicados por título.
    df = df.drop_duplicates(subset=["titles"])

    # Nos quedamos con un subconjunto para que el índice sea liviano.
    df = df.head(max_docs).reset_index(drop=True)

    print(f"Corpus final: {len(df)} documentos.")
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = preparar_corpus()
    df.to_csv(CORPUS_CSV_PATH, index=False)
    print(f"Corpus guardado en: {CORPUS_CSV_PATH}")


if __name__ == "__main__":
    main()
