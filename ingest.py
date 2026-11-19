# Módulo de ingesta: lee los documentos, los parte en fragmentos (chunks)
# y los guarda en ChromaDB.
#
# Uso:
#   python ingest.py           -> indexa solo si la base está vacía
#   python ingest.py --force   -> borra la base y vuelve a indexar

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

load_dotenv()  # carga OPENAI_API_KEY desde el archivo .env


def cargar_documentos():
    """Lee todos los archivos .md y .txt de la carpeta data/."""
    documentos = []
    for archivo in sorted(Path(config.CARPETA_DATOS).iterdir()):
        if archivo.suffix in [".md", ".txt"]:
            texto = archivo.read_text(encoding="utf-8")
            # Guardamos el nombre del archivo para poder citarlo después
            documentos.append(Document(page_content=texto, metadata={"source": archivo.name}))
    return documentos


def partir_en_chunks(documentos):
    """Parte los documentos en fragmentos de ~500 tokens con 50 de solapamiento."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # cuenta tokens igual que los modelos de OpenAI
        chunk_size=config.TAMANO_CHUNK,
        chunk_overlap=config.SOLAPAMIENTO,
    )
    return splitter.split_documents(documentos)


def obtener_vectorstore():
    """Abre (o crea) la colección de ChromaDB guardada en ./vectorstore."""
    embeddings = OpenAIEmbeddings(model=config.MODELO_EMBEDDINGS)
    return Chroma(
        collection_name=config.NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=config.CARPETA_VECTORSTORE,
    )


def crear_base_vectorial(forzar=False):
    """Indexa los documentos, pero solo si la base todavía no existe."""
    if forzar and Path(config.CARPETA_VECTORSTORE).exists():
        shutil.rmtree(config.CARPETA_VECTORSTORE)
        print("Base anterior borrada.")

    vectorstore = obtener_vectorstore()

    # Verificamos si ya hay datos para no indexar dos veces (ahorra tiempo y dinero)
    cantidad = len(vectorstore.get()["ids"])
    if cantidad > 0:
        print(f"La base ya existe con {cantidad} fragmentos. No se vuelve a indexar.")
        return vectorstore

    documentos = cargar_documentos()
    chunks = partir_en_chunks(documentos)
    print(f"Se leyeron {len(documentos)} documentos y se generaron {len(chunks)} fragmentos.")

    vectorstore.add_documents(chunks)
    print(f"Fragmentos guardados en ./{config.CARPETA_VECTORSTORE}")
    return vectorstore


if __name__ == "__main__":
    crear_base_vectorial(forzar="--force" in sys.argv)
