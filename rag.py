# Cadena RAG: recupera fragmentos de ChromaDB y genera una respuesta
# usando SOLO esa información.

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from ingest import obtener_vectorstore

load_dotenv()


# 1. Formato de la respuesta (modelo Pydantic)
class RespuestaRAG(BaseModel):
    respuesta: str = Field(description="Respuesta a la pregunta, o 'No lo sé' si no está en el contexto")
    fuentes: list[str] = Field(description="Nombres de los archivos usados para responder. Vacía si la respuesta es 'No lo sé'")


parser = PydanticOutputParser(pydantic_object=RespuestaRAG)


# 2. Retriever: convierte una consulta en embedding y trae los TOP_K fragmentos más parecidos
vectorstore = obtener_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})


def formatear_documentos(documentos):
    """Une los fragmentos recuperados en un solo texto, indicando de qué archivo viene cada uno."""
    partes = []
    for doc in documentos:
        partes.append(f"[Fuente: {doc.metadata['source']}]\n{doc.page_content}")
    return "\n\n---\n\n".join(partes)


# 3. Prompt: funciona como "filtro de veracidad"
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente técnico que responde preguntas sobre RAG (Retrieval-Augmented Generation).\n"
     "Reglas:\n"
     "1. Responde SOLO con la información del CONTEXTO. No uses tu conocimiento propio.\n"
     "2. Si la respuesta no está en el CONTEXTO, responde exactamente: \"No lo sé\" y deja la lista de fuentes vacía.\n"
     "3. Responde siempre en español.\n\n"
     "{format_instructions}"),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(format_instructions=parser.get_format_instructions())


# 4. Modelo de lenguaje (temperature=0 para respuestas más estables)
llm = ChatOpenAI(model=config.MODELO_LLM, temperature=0)


# 5. Multi-query en dos pasos.
#    Una pregunta que compara varias cosas ("¿en qué se diferencian A, B y C?") no se parece
#    a ninguna sección en particular, así que el buscador trae sobre todo introducciones.
#    Paso 1: buscamos con la pregunta original. Esos fragmentos suelen nombrar las partes
#            (por ejemplo, "RAG Ingenuo, RAG Avanzado y RAG Modular").
#    Paso 2: con ese contexto, el LLM escribe una subpregunta por cada parte y buscamos con cada una.
class Subpreguntas(BaseModel):
    subpreguntas: list[str] = Field(description="Subpreguntas cortas y autónomas")


prompt_subpreguntas = ChatPromptTemplate.from_messages([
    ("system",
     "Descompones preguntas para un buscador semántico.\n"
     "Escribe hasta {n} subpreguntas cortas, en español, que juntas cubran la PREGUNTA. "
     "Si la pregunta compara o enumera varias cosas, escribe una subpregunta por cada una "
     "y nómbrala con el nombre exacto que usa el CONTEXTO. "
     "Si la pregunta ya es simple, devuelve solo una. "
     "Las subpreguntas solo se usan para buscar: no las respondas."),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(n=config.NUM_SUBPREGUNTAS)

generar_subpreguntas = prompt_subpreguntas | llm.with_structured_output(Subpreguntas)


async def recuperar_multiconsulta(pregunta: str):
    """Busca con la pregunta original y con subpreguntas generadas a partir de esos resultados.

    Devuelve los fragmentos de todas las búsquedas, sin repetir y en orden de aparición.
    """
    iniciales = await retriever.ainvoke(pregunta)
    resultado = await generar_subpreguntas.ainvoke(
        {"pregunta": pregunta, "contexto": formatear_documentos(iniciales)}
    )
    subpreguntas = resultado.subpreguntas[: config.NUM_SUBPREGUNTAS]
    listas = [iniciales] + await retriever.abatch(subpreguntas)  # búsquedas en paralelo

    vistos, documentos = set(), []
    for lista in listas:
        for doc in lista:
            if doc.page_content not in vistos:
                vistos.add(doc.page_content)
                documentos.append(doc)
    return documentos


# 6. Cadena LCEL: pregunta -> multi-query -> prompt -> LLM -> parser
cadena_rag = (
    {
        "contexto": RunnableLambda(recuperar_multiconsulta) | formatear_documentos,
        "pregunta": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)


async def get_rag_response(query: str) -> RespuestaRAG:
    """Responde una pregunta de forma asíncrona.

    a. Busca en ChromaDB con la pregunta y con subpreguntas generadas a partir de esos resultados (multi-query).
    b. Arma el prompt con esos fragmentos.
    c. Llama al LLM de forma asíncrona (ainvoke).
    d. Convierte la respuesta al modelo Pydantic RespuestaRAG.
    """
    return await cadena_rag.ainvoke(query)
