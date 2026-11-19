# Sistema RAG local con LangChain y ChromaDB

Pre-entrega 3. Sistema de recuperación semántica local (RAG) que responde preguntas usando **solo** la información de unos documentos. Si la respuesta no está en los documentos, responde **"No lo sé"**.

## Cómo funciona

```
INGESTA (una sola vez)
data/*.md -> fragmentos de 500 tokens (overlap 50) -> embeddings -> ChromaDB (./vectorstore)

CONSULTA (cada pregunta)
pregunta -> embedding -> 4 fragmentos más parecidos -> prompt -> gpt-4o-mini -> respuesta + fuentes
```

- **Embeddings:** `text-embedding-3-small`. Es el mismo modelo para indexar y para consultar, y está definido en `config.py`.
- **LLM:** `gpt-4o-mini`, con `temperature=0`.
- **Top K:** 4 fragmentos.
- **Salida:** un modelo Pydantic `RespuestaRAG` con dos campos: `respuesta` y `fuentes`.

## Estructura

| Archivo | Qué hace |
|---|---|
| `data/` | Dataset: el paper *"Retrieval-Augmented Generation for Large Language Models: A Survey"* (Gao et al., 2024) traducido al español, en 4 archivos `.md` |
| `config.py` | Configuración: modelos, rutas, tamaño de los fragmentos y top K |
| `ingest.py` | Lee `data/`, parte el texto en fragmentos y lo guarda en ChromaDB. Si la base ya existe, no vuelve a indexar |
| `rag.py` | Retriever, prompt, cadena LCEL y la función asíncrona `get_rag_response(query)` |
| `main.py` | Ejecuta las dos pruebas: una pregunta válida y una pregunta trampa |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Crea un archivo `.env` a partir del ejemplo y coloca tu API key de OpenAI:

```bash
cp .env.example .env
```

El archivo `.env` está en `.gitignore`, así que la key nunca se sube a GitHub.

## Ejecución

```bash
# 1. Crear la base vectorial (solo la primera vez)
python ingest.py

# 2. Ejecutar las dos pruebas
python main.py
```

Para reindexar desde cero, por ejemplo si cambiaste los documentos:

```bash
python ingest.py --force
```

## Pruebas

| Prueba | Pregunta | Resultado esperado |
|---|---|---|
| 1. Respuesta en los documentos | ¿Cuáles son los tres paradigmas de RAG y en qué se diferencian? | Explica el RAG Ingenuo, el Avanzado y el Modular, y cita las fuentes |
| 2. Pregunta trampa | ¿Cuánto cuesta por millón de tokens usar el modelo gpt-4o-mini? | "No lo sé", con la lista de fuentes vacía |

La pregunta trampa es a propósito algo que el modelo podría "saber" por su entrenamiento, pero que no aparece en los documentos. Así se comprueba que el prompt lo obliga a responder solo con el contexto.

## Usar la función desde otro código

```python
import asyncio
from rag import get_rag_response

resultado = asyncio.run(get_rag_response("¿Qué es el problema 'Lost in the middle'?"))
print(resultado.respuesta)
print(resultado.fuentes)
```
