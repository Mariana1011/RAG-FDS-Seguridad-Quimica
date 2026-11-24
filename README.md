# Sistema RAG para Fichas de Datos de Seguridad (FDS)

## Descripción

Sistema de generación aumentada por recuperación (RAG) para consultar información técnica contenida en 101 Fichas de Datos de Seguridad de productos químicos.

El sistema recupera fragmentos del corpus documental y utiliza un modelo de lenguaje local para generar respuestas apoyadas en las fuentes encontradas.

## Tecnologías

- Python 3
- Amazon SageMaker: entorno de ejecución
- Sentence Transformers: embeddings multilingües
- FAISS: búsqueda vectorial
- Scikit-learn y TF-IDF: búsqueda léxica
- Ollama y Llama 3.2 3B: generación de respuestas
- Qwen2.5 7B y RAGAS: evaluación
- JSONL y Markdown: almacenamiento documental y resultados

## Estructura principal

RAG_FDS/
- corpus/
  - corpus_markdown/: documentos procesados
  - chunks.jsonl: fragmentos originales
  - chunks_depurados.jsonl: fragmentos depurados
- indices/: índices vectoriales y léxicos
- src/
  - rag_general.py: sistema RAG principal
  - evaluar_ragas_total.py: evaluación RAGAS
- resultados/
  - preguntas_fds_12_referencias.json
  - evaluacion_rag_24.jsonl
  - ragas_evaluacion_completa.jsonl
- README.md

## Preparación del entorno

Se requiere Python, las bibliotecas de recuperación y Ollama con el modelo llama3.2:3b disponible.

Activar el entorno virtual existente:

    source .venv/bin/activate

Comprobar los modelos instalados:

    curl http://localhost:11434/api/tags

## Ejecución

Desde la carpeta raíz RAG_FDS:

Consultar solamente las fuentes recuperadas:

    python src/rag_general.py --solo-busqueda "¿Qué dice la FDS 51 sobre el xileno?" --k 3

Generar una respuesta:

    python src/rag_general.py "¿Qué dice la FDS 51 sobre el xileno?" --k 3

Comparar con seis fragmentos:

    python src/rag_general.py "¿Qué dice la FDS 51 sobre el xileno?" --k 6

Las fuentes incluyen identificadores de FDS, páginas y fragmentos recuperados.

## Metodología de evaluación

Se utilizaron 12 preguntas:
- 6 de respuesta directa.
- 2 de integración de secciones.
- 2 comparativas.
- 2 sin respuesta disponible en las FDS.

Cada pregunta se ejecutó con top-k=3 y top-k=6, generando 24 respuestas.

Se evaluaron 10 preguntas respondibles mediante las métricas de RAGAS:
- Faithfulness
- Context Precision
- Context Recall

El modelo evaluador fue Qwen2.5 7B, ejecutado localmente mediante Ollama.

Se obtuvieron 59 de 60 mediciones previstas. La medición faltante fue Faithfulness para P06 con k=6, debido a errores persistentes de tiempo de espera.

Los resultados originales están disponibles en resultados/.

## Limitaciones

- Algunas secciones y metadatos del corpus proceden de extracción automática y requieren verificación documental.
- La calidad de recuperación depende de la estructura y legibilidad de las FDS.
- El modelo de lenguaje puede producir respuestas incompletas o incorrectas.
- Las puntuaciones de RAGAS dependen del evaluador local utilizado.
- La ejecución de modelos sin GPU puede presentar latencias elevadas.
- El sistema no sustituye la consulta de la FDS original ni la valoración de un profesional de seguridad química.

## Reproducibilidad

Los índices y resultados se conservan para evitar repetir el procesamiento del corpus y los experimentos.

Para reproducir exactamente las evaluaciones se requieren las mismas versiones de bibliotecas, modelos, preguntas, referencias, índices y parámetros de generación.

El archivo de resultados incluye las respuestas correspondientes a ambos valores de k.
