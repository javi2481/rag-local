# Configuración compartida por todo el proyecto.
# Está en un solo lugar para usar SIEMPRE el mismo modelo de embeddings
# al indexar (ingest.py) y al consultar (rag.py).

CARPETA_DATOS = "data"
CARPETA_VECTORSTORE = "vectorstore"
NOMBRE_COLECCION = "paper_rag"

MODELO_EMBEDDINGS = "text-embedding-3-small"
MODELO_LLM = "gpt-4o-mini"

TAMANO_CHUNK = 500      # tokens por fragmento
SOLAPAMIENTO = 50       # tokens que se repiten entre fragmentos vecinos
TOP_K = 4               # fragmentos que trae cada búsqueda (pregunta original y cada subpregunta)
