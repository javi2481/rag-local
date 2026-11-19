# RAG para Modelos de Lenguaje Grandes: Evaluación, desafíos futuros y conclusión

> Traducción al español (adaptada) de las secciones VI, VII y VIII del artículo "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al., arXiv:2312.10997v5, 2024). Se omitieron las tablas de tareas y conjuntos de datos, y los números de las referencias bibliográficas. La tabla de marcos de evaluación se transcribió como lista.

## VI. Tareas y evaluación

El rápido avance y la creciente adopción de RAG en el campo del procesamiento de lenguaje natural pusieron la evaluación de los modelos RAG en el centro de la investigación de la comunidad de LLM. El objetivo principal de esta evaluación es comprender y optimizar el rendimiento de los modelos RAG en diversos escenarios de aplicación. Esta sección presenta las principales tareas posteriores de RAG, los conjuntos de datos y cómo evaluar los sistemas RAG.

### A. Tareas posteriores (downstream tasks)

La tarea central de RAG sigue siendo la de Preguntas y Respuestas (QA, Question Answering), que incluye las preguntas tradicionales de un solo salto y de múltiples saltos (single-hop y multi-hop), las de opción múltiple, las de dominios específicos y los escenarios de respuestas largas (long-form) adecuados para RAG. Además de QA, RAG se está extendiendo continuamente a muchas otras tareas, como la Extracción de Información (IE), la generación de diálogos, la búsqueda de código, entre otras. En total, el artículo resume 26 tareas y cerca de 50 conjuntos de datos.

### B. Objetivos de la evaluación

Históricamente, la evaluación de los modelos RAG se centró en su desempeño en tareas posteriores específicas, usando las métricas establecidas para cada tarea. Por ejemplo, la evaluación de preguntas y respuestas puede basarse en las métricas EM (Exact Match, coincidencia exacta) y F1, mientras que las tareas de verificación de hechos (fact-checking) suelen usar la exactitud (Accuracy) como métrica principal. Las métricas BLEU y ROUGE también se usan con frecuencia para evaluar la calidad de las respuestas. Herramientas como RALLE, diseñada para la evaluación automática de aplicaciones RAG, también basan sus evaluaciones en estas métricas específicas de cada tarea. Aun así, hay pocas investigaciones dedicadas a evaluar las características propias de los modelos RAG. Los principales objetivos de evaluación son:

**Calidad de la recuperación.** Evaluar la calidad de la recuperación es crucial para determinar la efectividad del contexto obtenido por el componente recuperador. Se usan métricas estándar de los motores de búsqueda, los sistemas de recomendación y los sistemas de recuperación de información para medir el rendimiento del módulo de recuperación. Las métricas más comunes son Hit Rate (tasa de acierto), MRR (Mean Reciprocal Rank, rango recíproco medio) y NDCG (Normalized Discounted Cumulative Gain, ganancia acumulada descontada normalizada).

**Calidad de la generación.** La evaluación de la calidad de la generación se centra en la capacidad del generador para sintetizar respuestas coherentes y relevantes a partir del contexto recuperado. Esta evaluación puede dividirse según el tipo de contenido: sin etiquetar y etiquetado. Para el contenido sin etiquetar, la evaluación abarca la fidelidad (faithfulness), la relevancia y la inocuidad (que no sea dañino) de las respuestas generadas. Para el contenido etiquetado, el foco está en la exactitud de la información producida por el modelo. Tanto la calidad de la recuperación como la de la generación pueden evaluarse con métodos manuales o automáticos.

### C. Aspectos de la evaluación

Las prácticas actuales de evaluación de modelos RAG enfatizan tres puntajes de calidad principales y cuatro capacidades esenciales, que en conjunto informan la evaluación de los dos objetivos principales del modelo RAG: la recuperación y la generación.

**1) Puntajes de calidad.** Incluyen la relevancia del contexto, la fidelidad de la respuesta y la relevancia de la respuesta. Estos puntajes evalúan la eficiencia del modelo RAG desde distintas perspectivas del proceso de recuperación y generación.

- **Relevancia del contexto (Context Relevance):** evalúa la precisión y especificidad del contexto recuperado, asegurando su relevancia y minimizando los costos de procesar contenido innecesario.
- **Fidelidad de la respuesta (Answer Faithfulness):** asegura que las respuestas generadas se mantengan fieles al contexto recuperado, con coherencia y sin contradicciones.
- **Relevancia de la respuesta (Answer Relevance):** exige que las respuestas generadas sean directamente pertinentes a las preguntas planteadas y resuelvan la consulta central.

**2) Capacidades requeridas.** La evaluación de RAG también abarca cuatro capacidades que indican su adaptabilidad y eficiencia: robustez frente al ruido, rechazo negativo, integración de información y robustez contrafáctica. Estas capacidades son fundamentales para el rendimiento del modelo frente a distintos desafíos y escenarios complejos, e influyen en los puntajes de calidad.

- **Robustez frente al ruido (Noise Robustness):** evalúa la capacidad del modelo para manejar documentos ruidosos, es decir, documentos relacionados con la pregunta pero que no contienen información sustancial.
- **Rechazo negativo (Negative Rejection):** evalúa el criterio del modelo para abstenerse de responder cuando los documentos recuperados no contienen el conocimiento necesario para responder la pregunta.
- **Integración de información (Information Integration):** evalúa la habilidad del modelo para sintetizar información de varios documentos y responder preguntas complejas.
- **Robustez contrafáctica (Counterfactual Robustness):** pone a prueba la capacidad del modelo para reconocer e ignorar errores conocidos dentro de los documentos, incluso cuando se le advierte sobre posible desinformación.

La relevancia del contexto y la robustez frente al ruido son importantes para evaluar la calidad de la recuperación, mientras que la fidelidad de la respuesta, la relevancia de la respuesta, el rechazo negativo, la integración de información y la robustez contrafáctica son importantes para evaluar la calidad de la generación.

### D. Benchmarks y herramientas de evaluación

Se propusieron una serie de benchmarks y herramientas para facilitar la evaluación de RAG. Estos instrumentos ofrecen métricas cuantitativas que no solo miden el rendimiento de los modelos RAG, sino que también ayudan a comprender sus capacidades en los distintos aspectos de evaluación. Benchmarks destacados como RGB, RECALL y CRUD se enfocan en evaluar las capacidades esenciales de los modelos RAG. Al mismo tiempo, herramientas automatizadas de vanguardia como RAGAS, ARES y TruLens usan LLM para juzgar los puntajes de calidad. En conjunto forman un marco sólido para la evaluación sistemática de modelos RAG. El resumen de los marcos de evaluación es el siguiente:

- **RGB (benchmark).** Objetivos: calidad de la recuperación y de la generación. Aspectos: robustez frente al ruido, rechazo negativo, integración de información y robustez contrafáctica. Métricas: exactitud (Accuracy), EM, exactitud y exactitud, respectivamente.
- **RECALL (benchmark).** Objetivo: calidad de la generación. Aspecto: robustez contrafáctica. Métrica: R-Rate (tasa de reaparición, Reappearance Rate).
- **RAGAS (herramienta).** Objetivos: calidad de la recuperación y de la generación. Aspectos: relevancia del contexto, fidelidad y relevancia de la respuesta. Métricas: métricas personalizadas para los dos primeros y similitud coseno para la relevancia de la respuesta.
- **ARES (herramienta).** Objetivos: calidad de la recuperación y de la generación. Aspectos: relevancia del contexto, fidelidad y relevancia de la respuesta. Métrica: exactitud en los tres aspectos.
- **TruLens (herramienta).** Objetivos: calidad de la recuperación y de la generación. Aspectos: relevancia del contexto, fidelidad y relevancia de la respuesta. Métricas: personalizadas.
- **CRUD (benchmark).** Objetivos: calidad de la recuperación y de la generación. Aspectos: generación creativa, preguntas y respuestas intensivas en conocimiento, corrección de errores y resumen. Métricas: BLEU, ROUGE-L, BertScore y RAGQuestEval.

## VII. Discusión y perspectivas futuras

A pesar del progreso considerable de la tecnología RAG, persisten varios desafíos que requieren una investigación profunda. Esta sección presenta los desafíos actuales y las futuras direcciones de investigación de RAG.

### A. RAG frente a contexto largo

Con el avance de la investigación, la ventana de contexto de los LLM se amplía continuamente. Hoy los LLM pueden manejar sin esfuerzo contextos de más de 200.000 tokens. Esto significa que las preguntas sobre documentos largos, que antes dependían de RAG, ahora pueden resolverse incorporando el documento completo directamente en el prompt. Esto generó discusiones sobre si RAG sigue siendo necesario cuando los LLM no están limitados por el contexto.

En realidad, RAG sigue teniendo un papel irreemplazable. Por un lado, darle al LLM una gran cantidad de contexto de una sola vez afecta significativamente su velocidad de inferencia, mientras que la recuperación por fragmentos y la entrega bajo demanda mejoran mucho la eficiencia operativa. Por otro lado, la generación basada en RAG puede ubicar rápidamente las referencias originales para que los usuarios verifiquen las respuestas generadas. Todo el proceso de recuperación y razonamiento es observable, mientras que la generación que depende solo de un contexto largo sigue siendo una caja negra. A la inversa, la ampliación del contexto abre nuevas oportunidades para RAG, permitiéndole abordar problemas más complejos y preguntas integradoras o de resumen que requieren leer una gran cantidad de material. Desarrollar nuevos métodos RAG en el contexto de ventanas súper largas es una de las tendencias futuras de investigación.

### B. Robustez de RAG

La presencia de ruido o de información contradictoria durante la recuperación puede perjudicar la calidad de las respuestas de RAG. Esta situación se describe con la frase "la desinformación puede ser peor que la falta de información". Mejorar la resistencia de RAG frente a entradas adversarias o contrafácticas está ganando impulso en la investigación y se convirtió en una métrica clave de rendimiento. Cuconasu y colaboradores analizaron qué tipo de documentos conviene recuperar, evaluando la relevancia de los documentos respecto del prompt, su posición y la cantidad incluida en el contexto. Sus resultados muestran que incluir documentos irrelevantes puede, inesperadamente, aumentar la exactitud en más de un 30%, lo que contradice la suposición inicial de que reduciría la calidad. Estos resultados subrayan la importancia de desarrollar estrategias especializadas para integrar la recuperación con los modelos de generación, y la necesidad de seguir investigando la robustez de RAG.

### C. Enfoques híbridos

Combinar RAG con ajuste fino se está convirtiendo en una estrategia líder. Determinar la integración óptima de RAG y ajuste fino (de forma secuencial, alternada o mediante entrenamiento conjunto de extremo a extremo) y cómo aprovechar las ventajas tanto del conocimiento parametrizado como del no parametrizado son áreas listas para explorar. Otra tendencia es introducir en RAG modelos de lenguaje pequeños (SLM) con funciones específicas, ajustados con los resultados del propio sistema RAG. Por ejemplo, CRAG entrena un evaluador de recuperación liviano que valora la calidad general de los documentos recuperados para una consulta y, según el nivel de confianza, dispara distintas acciones de recuperación de conocimiento.

### D. Leyes de escalamiento de RAG

Los modelos RAG de extremo a extremo y los modelos preentrenados basados en RAG siguen siendo un foco de los investigadores actuales. La cantidad de parámetros de estos modelos es uno de los factores clave. Aunque las leyes de escalamiento (scaling laws) están establecidas para los LLM, su aplicabilidad a RAG sigue siendo incierta. Estudios iniciales como RETRO++ empezaron a abordar este tema, pero la cantidad de parámetros de los modelos RAG todavía está por detrás de la de los LLM. La posibilidad de una Ley de Escalamiento Inversa, en la que modelos más pequeños superan a los más grandes, es especialmente interesante y merece más investigación.

### E. RAG listo para producción

La practicidad de RAG y su alineación con los requisitos de ingeniería facilitaron su adopción. Sin embargo, mejorar la eficiencia de la recuperación, mejorar la exhaustividad (recall) de documentos en bases de conocimiento grandes y garantizar la seguridad de los datos (por ejemplo, evitar que los LLM revelen sin querer las fuentes de los documentos o sus metadatos) son desafíos de ingeniería críticos que todavía deben resolverse.

El desarrollo del ecosistema RAG está muy influido por la evolución de su stack tecnológico. Herramientas clave como LangChain y LlamaIndex ganaron popularidad rápidamente con la aparición de ChatGPT, ofreciendo amplias APIs relacionadas con RAG y volviéndose esenciales en el mundo de los LLM. El stack tecnológico emergente, aunque no es tan rico en funcionalidades como LangChain y LlamaIndex, se destaca por sus productos especializados. Por ejemplo, Flowise AI prioriza un enfoque low-code, que permite desplegar aplicaciones de IA, incluido RAG, mediante una interfaz amigable de arrastrar y soltar. Otras tecnologías como HayStack, Meltano y Cohere Coral también están ganando atención por sus aportes. Además de los proveedores enfocados en IA, los proveedores tradicionales de software y servicios en la nube están ampliando su oferta con servicios centrados en RAG. Verba, de Weaviate, está diseñado para aplicaciones de asistente personal, mientras que Kendra, de Amazon, ofrece servicios de búsqueda empresarial inteligente que permiten explorar distintos repositorios de contenido mediante conectores integrados.

En el desarrollo de la tecnología RAG hay una clara tendencia hacia distintas direcciones de especialización: 1) Personalización: adaptar RAG a requisitos específicos. 2) Simplificación: hacer que RAG sea más fácil de usar para reducir la curva de aprendizaje inicial. 3) Especialización: optimizar RAG para servir mejor en entornos de producción.

El crecimiento mutuo de los modelos RAG y sus stacks tecnológicos es evidente: los avances tecnológicos establecen continuamente nuevos estándares para la infraestructura existente y, a su vez, las mejoras del stack impulsan el desarrollo de las capacidades de RAG. Los conjuntos de herramientas de RAG están convergiendo en un stack tecnológico base, que sienta las bases para aplicaciones empresariales avanzadas. Sin embargo, el concepto de una plataforma totalmente integrada e integral todavía pertenece al futuro y requiere más innovación y desarrollo.

### F. RAG multimodal

RAG superó sus límites iniciales de preguntas y respuestas basadas en texto y ahora abarca una gran variedad de datos de distintas modalidades. Esta expansión dio lugar a modelos multimodales innovadores que integran conceptos de RAG en varios dominios:

- **Imagen.** RA-CM3 es un modelo multimodal pionero que recupera y genera tanto texto como imágenes. BLIP-2 aprovecha codificadores de imagen congelados junto con LLM para un preentrenamiento visión-lenguaje eficiente, lo que permite conversiones de imagen a texto zero-shot. El método "Visualizar antes de escribir" (Visualize Before You Write) usa la generación de imágenes para guiar la generación de texto del modelo de lenguaje, con resultados prometedores en tareas de generación de texto abierta.
- **Audio y video.** El método GSS recupera y une fragmentos de audio para convertir datos traducidos automáticamente en datos de traducción de voz. UEOP marca un avance significativo en el reconocimiento automático de voz (ASR) de extremo a extremo al incorporar estrategias externas y offline para la conversión de voz a texto. Además, la fusión de atención basada en KNN aprovecha embeddings de audio y embeddings de texto semánticamente relacionados para refinar el ASR, acelerando la adaptación a nuevos dominios. Vid2Seq amplía los modelos de lenguaje con marcadores temporales especializados, lo que facilita predecir los límites de los eventos y sus descripciones textuales dentro de una única secuencia de salida.
- **Código.** RBPS se destaca en tareas de aprendizaje a pequeña escala recuperando ejemplos de código alineados con los objetivos de los desarrolladores mediante codificación y análisis de frecuencia. Este enfoque demostró eficacia en tareas como la generación de aserciones de prueba y la reparación de programas. Para conocimiento estructurado, el método CoK primero extrae de un grafo de conocimiento los hechos pertinentes a la consulta y luego los integra como pistas dentro de la entrada, mejorando el rendimiento en tareas de preguntas y respuestas sobre grafos de conocimiento.

## VIII. Conclusión

El resumen de este artículo enfatiza el avance significativo de RAG para mejorar las capacidades de los LLM, al integrar el conocimiento parametrizado de los modelos de lenguaje con los amplios datos no parametrizados de bases de conocimiento externas. La revisión muestra la evolución de las tecnologías RAG y su aplicación en muchas tareas diferentes. El análisis describe tres paradigmas de desarrollo dentro del marco RAG: RAG Ingenuo, RAG Avanzado y RAG Modular, cada uno de los cuales representa una mejora progresiva respecto de sus predecesores. La integración técnica de RAG con otras metodologías de IA, como el ajuste fino y el aprendizaje por refuerzo, amplió aún más sus capacidades. A pesar del progreso de la tecnología RAG, existen oportunidades de investigación para mejorar su robustez y su capacidad para manejar contextos extensos.

El alcance de las aplicaciones de RAG se está expandiendo hacia dominios multimodales, adaptando sus principios para interpretar y procesar formas de datos diversas como imágenes, videos y código. Esta expansión destaca las importantes implicaciones prácticas de RAG para el despliegue de la IA, y atrae el interés de los sectores académico e industrial. El crecimiento del ecosistema RAG se evidencia en el aumento de las aplicaciones de IA centradas en RAG y en el desarrollo continuo de herramientas de soporte. A medida que se amplía el panorama de aplicaciones de RAG, es necesario refinar las metodologías de evaluación para acompañar su evolución. Asegurar evaluaciones de rendimiento precisas y representativas es crucial para capturar plenamente los aportes de RAG a la comunidad de investigación y desarrollo en inteligencia artificial.
