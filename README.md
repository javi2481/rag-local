# Sistema RAG local con LangChain y ChromaDB

**Un asistente que responde preguntas sobre un paper usando solo lo que dice el paper, y que dice "No lo sé" cuando la respuesta no está.**

Un sistema de **Retrieval-Augmented Generation (RAG)** que corre en tu máquina: indexa el survey *"Retrieval-Augmented Generation for Large Language Models: A Survey"* (Gao et al., 2024), traducido al español, en una base vectorial local (ChromaDB), y responde con `gpt-4o-mini` citando el archivo del que sacó cada dato.

**La consigna, en una línea:** armar un RAG local con LangChain y ChromaDB, con una función asíncrona `get_rag_response(query)` que devuelva un modelo Pydantic, y probarlo con una pregunta que está en los documentos y una pregunta trampa que no está.

| | |
|---|---|
| **Tipo** | Pre-entrega 3 |
| **Autor** | Rodolfo Javier Berrone |
| **Año** | 2026 |
| **Documentos** | El survey de RAG de Gao et al. (arXiv:2312.10997v5), traducido al español en 4 archivos Markdown |
| **Estado** | Las dos pruebas pasan de forma estable (3 de 3 corridas) |

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C)
![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-FF6446)
![Embeddings](https://img.shields.io/badge/embeddings-text--embedding--3--small-412991)
![LLM](https://img.shields.io/badge/LLM-gpt--4o--mini-412991)
![Pydantic](https://img.shields.io/badge/salida-Pydantic-E92063)

## Contenido

- [El proyecto en una imagen](#el-proyecto-en-una-imagen)
- [Cómo funciona](#cómo-funciona)
- [1. Los documentos](#1-los-documentos)
- [2. Ingesta: chunking por títulos](#2-ingesta-chunking-por-títulos)
- [3. Búsqueda: multi-query en dos pasos](#3-búsqueda-multi-query-en-dos-pasos)
- [4. Generación: el prompt como filtro de veracidad](#4-generación-el-prompt-como-filtro-de-veracidad)
- [5. Pruebas](#5-pruebas)
- [Qué aprendimos](#qué-aprendimos)
- [Costos](#costos)
- [Reproducir](#reproducir)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Usar la función desde otro código](#usar-la-función-desde-otro-código)
- [Glosario rápido](#glosario-rápido)
- [Créditos](#créditos)

## El proyecto en una imagen

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#16a34a"}}}}%%
xychart-beta
    title "Secciones de paradigmas (Ingenuo, Avanzado, Modular) que llegan al LLM"
    x-axis ["v1: TOP_K 4", "v2: TOP_K 6", "v3: + títulos", "v4: + MMR", "v5: multi-query"]
    y-axis "secciones (de 3)" 0 --> 3
    bar [0, 1, 1, 1, 3]
```

La primera versión respondía bien la pregunta de prueba, *"¿Cuáles son los tres paradigmas de RAG y en qué se diferencian?"*, pero **solo los nombraba**: no explicaba ninguna diferencia. El LLM no tenía la culpa. **Al contexto no llegaban las secciones que explican cada paradigma**, solo el resumen y la introducción, que los mencionan de pasada. Probamos cuatro cambios y medimos cuántas de las tres secciones llegaban al LLM. Subir `TOP_K`, pegar el título a cada fragmento y MMR ayudaron poco. Lo que resolvió el problema fue **buscar en dos pasos**: primero con la pregunta y después con una subpregunta por cada paradigma. Este README explica cada paso.

## Cómo funciona

```mermaid
flowchart TD
    subgraph S1["1. Documentos"]
        A[("data/*.md<br/>4 archivos · survey de RAG en español")]
    end

    subgraph S2["2. Ingesta (una sola vez)"]
        B["Separar por títulos<br/>#, ##, ###"]
        C["Partir cada sección<br/>500 tokens · overlap 50"]
        D["Pegar la ruta de títulos<br/>«II. Visión general > A. RAG Ingenuo»"]
        E[("ChromaDB · ./vectorstore<br/>59 fragmentos · text-embedding-3-small")]
    end

    subgraph S3["3. Búsqueda (cada pregunta)"]
        F(["Pregunta"])
        G["Paso 1: buscar con la pregunta<br/>top 4"]
        H["LLM escribe hasta 3 subpreguntas<br/>con los nombres del contexto"]
        I["Paso 2: buscar con cada subpregunta<br/>top 4 cada una · sin repetidos"]
    end

    subgraph S4["4. Generación"]
        J["Prompt de veracidad<br/>«solo el CONTEXTO; si no está: No lo sé»"]
        K["gpt-4o-mini · temperature 0"]
        L[/"RespuestaRAG<br/>respuesta + fuentes"/]
    end

    A --> B --> C --> D --> E
    F --> G
    E -.-> G
    G --> H --> I
    E -.-> I
    I --> J --> K --> L

    classDef docs fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef ingesta fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef busqueda fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef gen fill:#fce7f3,stroke:#db2777,color:#831843

    class A docs
    class B,C,D,E ingesta
    class F,G,H,I busqueda
    class J,K,L gen

    style S1 fill:#eff6ff,stroke:#2563eb,color:#1e3a8a
    style S2 fill:#f0fdf4,stroke:#16a34a,color:#14532d
    style S3 fill:#fffbeb,stroke:#d97706,color:#78350f
    style S4 fill:#fdf2f8,stroke:#db2777,color:#831843
```

| Etapa | Pregunta que responde | Resultado |
|---|---|---|
| **1. Documentos** (azul) | ¿Sobre qué textos respondemos? | El survey de RAG en 4 archivos Markdown, sin figuras ni tablas |
| **2. Ingesta** (verde) | ¿Cómo partimos y guardamos el texto? | 59 fragmentos, cada uno con la ruta de títulos de su sección |
| **3. Búsqueda** (amarillo) | ¿Qué fragmentos le pasamos al LLM? | Unos 10 fragmentos (≈ 4.000 tokens) de la pregunta y de hasta 3 subpreguntas |
| **4. Generación** (rosa) | ¿Cómo evitamos que invente? | Un prompt que solo deja usar el contexto y una salida Pydantic con fuentes |

## 1. Los documentos

El dataset es el survey de Gao et al. (arXiv:2312.10997v5, marzo de 2024), **traducido al español y adaptado** en 4 archivos. El PDF original está en la raíz del repo (`2312.10997v5.pdf`).

| Archivo | Qué cubre |
|---|---|
| `01_introduccion_y_paradigmas_rag.md` | Resumen, introducción y sección II: RAG Ingenuo, Avanzado y Modular, y RAG frente a fine-tuning |
| `02_recuperacion.md` | Sección III: fuentes, indexado, optimización de la consulta, embeddings y adaptadores |
| `03_generacion_y_aumentacion.md` | Secciones IV y V: post-recuperación, ajuste fino del generador y recuperación iterativa, recursiva y adaptativa |
| `04_evaluacion_y_futuro.md` | Secciones VI a VIII: tareas, métricas, benchmarks, desafíos, RAG multimodal y conclusión |

Elegimos este paper a propósito: **el sistema responde preguntas sobre RAG usando RAG**. Además, tiene una estructura de títulos muy clara (secciones con números romanos y subsecciones A, B, C…), y esa estructura terminó siendo clave para la búsqueda.

## 2. Ingesta: chunking por títulos

`ingest.py` lee `data/`, parte el texto en fragmentos y los guarda en ChromaDB. Si la base ya tiene datos, no vuelve a indexar, para ahorrar tiempo y dinero.

```mermaid
flowchart LR
    M[("Archivo .md")] --> T["MarkdownHeaderTextSplitter<br/>una sección por título"]
    T --> S["RecursiveCharacterTextSplitter<br/>500 tokens · overlap 50<br/>(cuenta tokens con cl100k_base)"]
    S --> P["Pega adelante la ruta:<br/>«Sección: II. Visión general de RAG ><br/>A. RAG Ingenuo (Naive RAG)»"]
    P --> V[("ChromaDB<br/>metadatos: source + seccion")]

    classDef fuente fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef paso fill:#f3f4f6,stroke:#6b7280,color:#1f2937
    classDef fin fill:#dcfce7,stroke:#16a34a,color:#14532d

    class M fuente
    class T,S,P paso
    class V fin
```

**Por qué pegar los títulos.** La primera versión partía el texto cada 500 tokens sin mirar la estructura. Un fragmento del medio de la sección "A. RAG Ingenuo" empezaba con *"**Indexado.** Comienza con la limpieza…"*: no decía en ningún lado que hablaba del RAG Ingenuo. El vector de ese fragmento no tenía forma de parecerse a una pregunta sobre paradigmas. Ahora cada fragmento empieza con su ruta de títulos (el `#` principal se omite porque es el título del archivo), y esa ruta se guarda también en el metadato `seccion`.

| | v1 — Tamaño fijo | v3 — Por títulos |
|---|---:|---:|
| Fragmentos | 46 | 59 |
| ¿Respeta las secciones? | No: un fragmento puede mezclar el final de una y el principio de otra | Sí: nunca cruza un título |
| ¿El fragmento sabe de qué sección es? | No | Sí: la ruta va en el texto y en los metadatos |

> **Si cambias la forma de partir el texto, hay que reindexar** con `python ingest.py --force`. La ingesta normal ve que la base ya tiene datos y no hace nada.

## 3. Búsqueda: multi-query en dos pasos

Este fue el problema más largo del proyecto. Para encontrarlo miramos **en qué puesto aparecía cada sección** cuando se buscaba con la pregunta de prueba:

| Sección | Puesto (de 59) |
|---|---:|
| II. Visión general de RAG (solo nombra los tres paradigmas) | 1 |
| I. Introducción | 2 |
| C. RAG Modular | 3 y 4 |
| **A. RAG Ingenuo** | **13** |
| **B. RAG Avanzado** | **28** |

Una pregunta que **compara** varias cosas se parece más a los textos que las *enumeran* (el resumen, la introducción) que a los que las *explican*. Por eso los cambios que probamos primero no alcanzaron:

| Versión | Cambio | Secciones A/B/C en el contexto | Por qué |
|---|---|---:|---|
| v1 | Tamaño fijo, `TOP_K = 4` | 0 | Entran resumen, introducción y texto del archivo 04 |
| v2 | `TOP_K = 6` | 1 | Los lugares extra se llenan con más texto de relleno |
| v3 | + chunking por títulos | 1 | C sube al puesto 3, pero A y B siguen lejos |
| v4 | + MMR (`fetch_k = 20`) | 1 | MMR elige entre los 20 primeros, y B está en el 28 |
| **v5** | **multi-query en dos pasos** | **3** | **Cada paradigma tiene su propia búsqueda** |

La solución fue **descomponer la pregunta**. Con una subpregunta por paradigma, cada sección queda en el primer puesto de su búsqueda.

```mermaid
sequenceDiagram
    autonumber
    actor U as Usuario
    participant R as rag.py
    participant C as ChromaDB
    participant L as gpt-4o-mini

    U->>R: ¿Cuáles son los tres paradigmas de RAG y en qué se diferencian?
    R->>C: buscar(pregunta) · top 4
    C-->>R: Visión general · Introducción · RAG Modular ×2
    R->>L: CONTEXTO + PREGUNTA → hasta 3 subpreguntas
    L-->>R: ¿Qué es el RAG Ingenuo…? · ¿…el RAG Avanzado…? · ¿…el RAG Modular…?
    loop una búsqueda por subpregunta
        R->>C: buscar(subpregunta) · top 4
    end
    C-->>R: + RAG Ingenuo · RAG Avanzado · …
    R->>R: agregar solo los nuevos → ≈ 10 fragmentos
    R->>L: prompt de veracidad + 10 fragmentos
    L-->>U: RespuestaRAG(respuesta, fuentes)
```

**Por qué en dos pasos y no en uno.** La primera versión del multi-query le pedía las subpreguntas al LLM directamente. Salían genéricas, como *"¿qué caracteriza al primer paradigma?"*: el modelo todavía no había leído el documento y no nombraba los paradigmas. Ahora primero se busca con la pregunta original y esos fragmentos se le pasan al LLM, con la instrucción de usar **los nombres exactos que aparecen en el contexto**. Como el resumen dice "RAG Ingenuo, RAG Avanzado y RAG Modular", las subpreguntas salen con esos nombres.

El LLM devuelve las subpreguntas como texto, una por línea. Después se busca con cada una en un `for`, y solo se agregan los fragmentos que todavía no estaban en la lista.

## 4. Generación: el prompt como filtro de veracidad

```mermaid
flowchart TD
    CTX["CONTEXTO<br/>fragmentos con [Fuente: archivo.md]"] --> P["Prompt de sistema<br/>1. solo el CONTEXTO, sin conocimiento propio<br/>2. si no está: exactamente «No lo sé» y fuentes vacías<br/>3. siempre en español"]
    Q(["PREGUNTA"]) --> P
    P --> L["gpt-4o-mini<br/>temperature = 0"]
    L --> PA["PydanticOutputParser"]
    PA --> OK["✅ respuesta + fuentes"]
    PA --> NS["🚫 «No lo sé» + []"]

    classDef entrada fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef paso fill:#f3f4f6,stroke:#6b7280,color:#1f2937
    classDef ok fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef no fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    class CTX,Q entrada
    class P,L,PA paso
    class OK ok
    class NS no
```

La salida es un modelo Pydantic:

```python
class RespuestaRAG(BaseModel):
    respuesta: str      # la respuesta, o "No lo sé"
    fuentes: list[str]  # archivos usados; vacía si la respuesta es "No lo sé"
```

Cada fragmento llega al LLM con su archivo de origen (`[Fuente: 01_introduccion_y_paradigmas_rag.md]`), así el modelo puede citar de dónde sacó cada dato. `temperature = 0` hace que las respuestas sean estables entre corridas.

## 5. Pruebas

`main.py` prepara la base (solo indexa si está vacía) y ejecuta las dos pruebas de la consigna.

| Prueba | Pregunta | Resultado esperado |
|---|---|---|
| 1. Respuesta en los documentos | ¿Cuáles son los tres paradigmas de RAG y en qué se diferencian? | Explica el RAG Ingenuo, el Avanzado y el Modular, y cita la fuente |
| 2. Pregunta trampa | ¿Cuánto cuesta por millón de tokens usar el modelo gpt-4o-mini? | "No lo sé", con la lista de fuentes vacía |

La pregunta trampa es, a propósito, algo que el modelo podría "saber" por su entrenamiento pero que no aparece en los documentos. Así se comprueba que el prompt lo obliga a responder solo con el contexto.

**Prueba 1, antes y después:**

| | v1 (tamaño fijo, `TOP_K = 4`) | v5 (títulos + multi-query) |
|---|---|---|
| Respuesta | *"Los tres paradigmas de RAG son: RAG Ingenuo, RAG Avanzado y RAG Modular. El RAG Ingenuo es la etapa inicial, mientras que el RAG Avanzado y el RAG Modular son respuestas a las limitaciones específicas del RAG Ingenuo…"* | **RAG Ingenuo:** la metodología más temprana, con indexado, recuperación y generación; tiene problemas de precisión y genera respuestas irrelevantes. **RAG Avanzado:** mejora la calidad de la recuperación con estrategias previas y posteriores a la recuperación. **RAG Modular:** más adaptable, con módulos especializados que se pueden combinar. |
| ¿Explica diferencias? | No: solo dice que uno vino después del otro | Sí: una característica concreta de cada uno |
| Fuentes | `01_introduccion_y_paradigmas_rag.md` | `01_introduccion_y_paradigmas_rag.md` y `04_evaluacion_y_futuro.md` (de este cita un fragmento sobre RAG en producción) |

**Prueba 2:** en todas las versiones respondió `"No lo sé"` con `fuentes = []`. El multi-query no la rompió: las subpreguntas (*"¿cuál es el costo por millón de tokens…?"*) traen fragmentos sobre embeddings y benchmarks, y ninguno tiene precios, así que el prompt de veracidad hace su trabajo.

Corrimos `main.py` **3 veces** con la versión final, y las dos pruebas dieron el mismo resultado en las tres.

## Qué aprendimos

1. **Si la respuesta es pobre, mirar primero qué llegó al contexto.** El LLM respondía bien con lo que recibía. El problema estaba en la búsqueda, y se vio recién cuando listamos los fragmentos recuperados y el puesto de cada sección.
2. **Más fragmentos no es mejor contexto.** Subir `TOP_K` de 4 a 6 sumó texto de relleno y casi nada útil.
3. **Un fragmento tiene que saber de qué habla.** Pegar la ruta de títulos mejoró la búsqueda, igual que pegar el nombre del autor en [rag-literatura-latinoamericana](https://github.com/javi2481/rag-literatura-latinoamericana). Pero no alcanzó, porque el problema era la forma de la pregunta.
4. **Las preguntas que comparan necesitan varias búsquedas.** Una comparación se parece al texto que enumera, no al que explica. Descomponerla en una subpregunta por elemento fue el cambio que resolvió el problema.
5. **Las subpreguntas tienen que salir del documento.** Sin contexto, el LLM escribía subpreguntas vagas. Con los primeros fragmentos delante, usa los nombres exactos del paper.

> **Ojo con leer estos resultados como definitivos.** Son 2 preguntas de prueba, una de ellas trampa. La mejora de la búsqueda se midió con una sola pregunta, contando cuántas de las tres secciones llegaban al LLM. No es una evaluación con un set de preguntas anotadas como `hit@k`.

**Qué haríamos distinto o seguiríamos:** armar un set de 10 a 20 preguntas con la sección esperada y medir `hit@k` de cada versión; probar un *reranker* sobre los fragmentos unidos; saltear el multi-query cuando la pregunta es simple, para ahorrar la llamada extra al LLM.

## Costos

Los dos modelos son los más baratos de OpenAI que alcanzan para esta tarea. Los precios cambian: estos son aproximados y conviene revisarlos en [openai.com/api/pricing](https://openai.com/api/pricing).

| Qué | Cuánto | Costo aproximado |
|---|---|---|
| Indexar (una sola vez) | 59 fragmentos, ≈ 18.000 tokens de embeddings | menos de USD 0,001 |
| Una pregunta: embeddings | 4 búsquedas (la pregunta y 3 subpreguntas) | despreciable |
| Una pregunta: subpreguntas | 1 llamada a `gpt-4o-mini` con ≈ 2.000 tokens | ≈ USD 0,0003 |
| Una pregunta: respuesta | 1 llamada a `gpt-4o-mini` con ≈ 4.000 tokens de contexto | ≈ USD 0,0007 |
| **Una corrida de `main.py`** | **2 preguntas** | **≈ USD 0,002** |

Con 1 dólar alcanza para unas 500 corridas de `main.py`.

## Reproducir

**Necesitas:** Python 3.11 y una API key de OpenAI con créditos.

```bash
git clone https://github.com/javi2481/rag-local.git
cd rag-local
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # y pega tu API key en .env
```

El archivo `.env` está en `.gitignore`, así que la key nunca se sube a GitHub.

```bash
python main.py                   # indexa si hace falta y corre las dos pruebas
```

```mermaid
flowchart TD
    S(["python main.py"]) --> Q{"¿./vectorstore<br/>tiene fragmentos?"}
    Q -->|"No (primera vez)"| I["Indexa data/<br/>59 fragmentos · ≈ 18.000 tokens"]
    Q -->|"Sí"| N["No vuelve a indexar"]
    I --> P["Prueba 1 y prueba 2"]
    N --> P
    F(["python ingest.py --force"]) --> B["Borra ./vectorstore"] --> I

    classDef inicio fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef paga fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef gratis fill:#dcfce7,stroke:#16a34a,color:#14532d

    class S,F,Q inicio
    class I,B paga
    class N,P gratis
```

| Comando | Cuándo usarlo |
|---|---|
| `python main.py` | Siempre: indexa solo si la base está vacía y corre las pruebas |
| `python ingest.py` | Para indexar sin correr las pruebas |
| `python ingest.py --force` | Cuando cambias los documentos de `data/`, el chunking o el modelo de embeddings |

Todos los parámetros están en `config.py`:

| Parámetro | Valor | Qué controla |
|---|---|---|
| `MODELO_EMBEDDINGS` | `text-embedding-3-small` | Modelo para indexar **y** para buscar: tiene que ser el mismo en los dos |
| `MODELO_LLM` | `gpt-4o-mini` | Modelo que genera las subpreguntas y la respuesta |
| `TAMANO_CHUNK` / `SOLAPAMIENTO` | 500 / 50 tokens | Tamaño de cada fragmento y cuánto se repite con el siguiente |
| `TOP_K` | 4 | Fragmentos que trae **cada** búsqueda |

## Estructura del repositorio

```text
.
├── data/                                  # el dataset: el survey de RAG en español
│   ├── 01_introduccion_y_paradigmas_rag.md
│   ├── 02_recuperacion.md
│   ├── 03_generacion_y_aumentacion.md
│   └── 04_evaluacion_y_futuro.md
├── 2312.10997v5.pdf                       # el paper original, en inglés
├── config.py                              # modelos, rutas, chunking y TOP_K
├── ingest.py                              # lee data/, parte por títulos y guarda en ChromaDB
├── rag.py                                 # búsqueda en dos pasos, prompt, cadena LCEL y get_rag_response()
├── main.py                                # las dos pruebas
├── requirements.txt
├── .env.example                           # plantilla para la API key
└── README.md
```

Al ejecutar se crea `vectorstore/`, la base de ChromaDB. Está en `.gitignore`, igual que `.env` y `.venv/`.

```mermaid
flowchart LR
    CFG["config.py"] --> ING["ingest.py"]
    CFG --> RAG["rag.py"]
    D[("data/*.md")] --> ING
    ENV[(".env<br/>OPENAI_API_KEY")] --> ING
    ENV --> RAG
    ING --> VS[("vectorstore/")]
    ING -->|"obtener_vectorstore()"| RAG
    VS --> RAG
    RAG -->|"get_rag_response()"| MAIN["main.py"]
    ING -->|"crear_base_vectorial()"| MAIN

    classDef config fill:#f3f4f6,stroke:#6b7280,color:#1f2937
    classDef datos fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef codigo fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef salida fill:#dcfce7,stroke:#16a34a,color:#14532d

    class CFG,ENV config
    class D datos
    class ING,RAG codigo
    class VS,MAIN salida
```

## Usar la función desde otro código

```python
import asyncio
from rag import get_rag_response

resultado = asyncio.run(get_rag_response("¿Qué es el problema 'Lost in the middle'?"))
print(resultado.respuesta)
print(resultado.fuentes)
```

La base tiene que estar creada antes: corre una vez `python ingest.py` o `python main.py`.

## Glosario rápido

| Término | Qué significa |
|---|---|
| **RAG** | Buscar primero los fragmentos relevantes de un corpus y pasárselos al LLM para que responda con ellos |
| **Chunk / fragmento** | Cada pedazo en que se parte un texto para buscarlo |
| **Overlap** | Tokens que se repiten entre un fragmento y el siguiente, para no perder información en el corte |
| **Embedding** | Vector de números que representa el significado de un texto; textos parecidos quedan cerca |
| **ChromaDB** | Base de datos local para guardar vectores y encontrar los más parecidos a uno dado |
| **TOP_K** | Cuántos fragmentos devuelve cada búsqueda |
| **MMR** | *Maximal Marginal Relevance*: elige fragmentos relevantes pero distintos entre sí |
| **Multi-query** | Buscar con varias versiones o partes de la pregunta y unir los resultados |
| **LCEL** | *LangChain Expression Language*: arma la cadena uniendo pasos con `\|` |
| **Prompt de veracidad** | Instrucción que obliga al LLM a responder solo con el contexto que recibe |
| **Alucinación** | Cuando el modelo inventa un dato con total seguridad |

## Créditos

- Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang y Haofen Wang, [*Retrieval-Augmented Generation for Large Language Models: A Survey*](https://arxiv.org/abs/2312.10997), arXiv:2312.10997v5, 2024: el paper que se usa como dataset, traducido y adaptado al español.
- [LangChain](https://github.com/langchain-ai/langchain): cadena LCEL, splitters y conectores.
- [ChromaDB](https://github.com/chroma-core/chroma): base vectorial local.
- [OpenAI](https://platform.openai.com/): `text-embedding-3-small` y `gpt-4o-mini`.
