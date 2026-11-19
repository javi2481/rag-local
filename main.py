# Script principal: prepara la base y ejecuta las dos pruebas pedidas.
#   1) Una pregunta cuya respuesta SÍ está en los documentos.
#   2) Una "pregunta trampa" cuya respuesta NO está (el modelo debe decir "No lo sé").

import asyncio

from ingest import crear_base_vectorial
from rag import get_rag_response

PREGUNTA_VALIDA = "¿Cuáles son los tres paradigmas de RAG y en qué se diferencian?"
PREGUNTA_TRAMPA = "¿Cuánto cuesta por millón de tokens usar el modelo gpt-4o-mini?"


def mostrar(titulo, pregunta, resultado):
    print("=" * 70)
    print(titulo)
    print("Pregunta: ", pregunta)
    print("Respuesta:", resultado.respuesta)
    print("Fuentes:  ", resultado.fuentes)


async def main():
    resultado = await get_rag_response(PREGUNTA_VALIDA)
    mostrar("PRUEBA 1: pregunta con respuesta en los documentos", PREGUNTA_VALIDA, resultado)

    resultado = await get_rag_response(PREGUNTA_TRAMPA)
    mostrar("PRUEBA 2: pregunta trampa (no está en los documentos)", PREGUNTA_TRAMPA, resultado)

    if "no lo sé" in resultado.respuesta.lower():
        print("\nOK: el modelo no alucinó en la pregunta trampa.")
    else:
        print("\nATENCIÓN: el modelo respondió algo en la pregunta trampa.")


if __name__ == "__main__":
    crear_base_vectorial()   # solo indexa si la base todavía no existe
    asyncio.run(main())
