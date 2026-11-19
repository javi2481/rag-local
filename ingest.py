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
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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
    """Parte los documentos en fragmentos de ~500 tokens con 50 de solapamiento.

    1. Separa cada documento por sus títulos de markdown (#, ##, ###).
    2. Parte cada sección en fragmentos de ~500 tokens.
    3. Agrega al inicio de cada fragmento la ruta de títulos de su sección,
       por ejemplo "II. Visión general de RAG > A. RAG Ingenuo (Naive RAG)".
       Así un fragmento del medio de una sección no pierde de qué habla,
       y el buscador lo puede relacionar con la pregunta.
    """
    niveles = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    separador_titulos = MarkdownHeaderTextSplitter(headers_to_split_on=niveles, strip_headers=True)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # cuenta tokens igual que los modelos de OpenAI
        chunk_size=config.TAMANO_CHUNK,
        chunk_overlap=config.SOLAPAMIENTO,
    )

    chunks = []
    for documento in documentos:
        secciones = separador_titulos.split_text(documento.page_content)
        for seccion in secciones:
            # Armamos la ruta con los títulos ## y ### (el # es el título del archivo)
            titulos = []
            if "h2" in seccion.metadata:
                titulos.append(seccion.metadata["h2"])
            if "h3" in seccion.metadata:
                titulos.append(seccion.metadata["h3"])
            ruta = " > ".join(titulos)

            for trozo in splitter.split_text(seccion.page_content):
                texto = f"Sección: {ruta}\n\n{trozo}"
                metadatos = {"source": documento.metadata["source"], "seccion": ruta}
                chunks.append(Document(page_content=texto, metadata=metadatos))
    return chunks


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
