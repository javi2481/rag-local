# Cadena RAG: recupera fragmentos de ChromaDB y genera una respuesta
# usando SOLO esa información.

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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


# 2. Retriever: convierte la pregunta en embedding y trae los TOP_K fragmentos más parecidos
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


# 5. Cadena LCEL: pregunta -> retriever -> prompt -> LLM -> parser
cadena_rag = (
    {
        "contexto": retriever | formatear_documentos,
        "pregunta": RunnablePassthrough(),
    }
    | prompt
    | llm
    | parser
)


async def get_rag_response(query: str) -> RespuestaRAG:
    """Responde una pregunta de forma asíncrona.

    a. Busca los fragmentos más similares en ChromaDB (retriever).
    b. Arma el prompt con esos fragmentos.
    c. Llama al LLM de forma asíncrona (ainvoke).
    d. Convierte la respuesta al modelo Pydantic RespuestaRAG.
    """
    return await cadena_rag.ainvoke(query)
