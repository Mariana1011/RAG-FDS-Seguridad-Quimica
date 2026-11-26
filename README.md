# Sistema RAG para Fichas de Datos de Seguridad (FDS)

## 1. Descripción

Sistema de generación aumentada por recuperación (Retrieval-Augmented Generation, RAG) desarrollado para consultar información de seguridad química contenida en 101 Fichas de Datos de Seguridad (FDS).

El proyecto integra procesamiento documental, recuperación híbrida, modelos de lenguaje ejecutados localmente, evaluación mediante RAGAS y una interfaz web interactiva.

Permite realizar consultas sobre peligros, composición, primeros auxilios, propiedades fisicoquímicas y otras secciones de las FDS, utilizando los fragmentos recuperados como contexto para generar respuestas.

## 2. Arquitectura

El sistema sigue el flujo:

PDF → Markdown → Fragmentación → Índices → Recuperación → LLM → Respuesta

### Procesamiento documental
- Corpus de 101 FDS.
- Conversión de PDF a Markdown.
- Conservación de recursos gráficos vinculados a los documentos.
- Identificación de secciones y páginas.
- Preparación de 6.604 fragmentos para recuperación.

### Recuperación híbrida
- FAISS: recuperación semántica mediante embeddings.
- TF-IDF: recuperación léxica mediante coincidencias de términos.
- Modelo de embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
- Dimensión de embeddings: 384.
- Índice vectorial: IndexFlatIP con vectores normalizados.
- Parámetro top-k configurable en 3 o 6.

La interfaz permite seleccionar recuperación híbrida FAISS + TF-IDF o recuperación exclusivamente léxica mediante TF-IDF.

### Generación
- Modelo generador: Llama 3.2 3B.
- Motor de inferencia: Ollama.
- Las respuestas se generan a partir del contexto documental recuperado.

### Infraestructura
- AWS Academy / Amazon SageMaker: desarrollo y ejecución.
- Gradio: interfaz web interactiva.
- ngrok: exposición de la interfaz para demostraciones.
- Ollama: ejecución local de los modelos de lenguaje.

## 3. Tecnologías

- Python
- PyMuPDF
- Sentence Transformers
- FAISS
- Scikit-learn
- TF-IDF
- Ollama
- Llama 3.2 3B
- Qwen2.5 7B
- RAGAS
- Gradio
- ngrok
- Amazon SageMaker

## 4. Estructura del repositorio

- `app.py`: interfaz web Gradio.
- `src/rag_general.py`: recuperación y generación RAG.
- `src/evaluar_ragas_total.py`: evaluación automatizada.
- `src/auditar_pdf_markdown.py`: comparación documental preliminar.
- `src/auditar_bloques_tecnicos.py`: auditoría de bloques técnicos.
- `src/extraer_catalogo_fds.py`: extracción de metadatos documentales.
- `corpus/corpus_markdown/`: FDS convertidas y recursos asociados.
- `corpus/inventario.csv`: inventario del corpus.
- `indices/`: índices de recuperación y metadatos de fragmentos.
- `resultados/`: preguntas, referencias, evaluaciones y auditorías.
- `requirements.txt`: dependencias de Python.

Los PDF originales se conservaron por separado durante el desarrollo. Los archivos Markdown, índices y evidencias permiten consultar la estructura documental procesada.

## 5. Instalación

### 5.1. Clonar el repositorio

```bash
git clone https://github.com/Mariana1011/RAG-FDS-Seguridad-Quimica.git
cd RAG-FDS-Seguridad-Quimica
```

### 5.2. Crear el entorno de Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.3. Descargar el modelo de embeddings

El modelo MiniLM se guarda localmente y no se incluye como archivo binario en GitHub.

Ejecutar con acceso a internet:

```bash
mkdir -p modelos

python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').save('modelos/minilm_multilingual')"
```

El sistema utiliza esta ruta local para cargar los embeddings y mantener compatibilidad con el índice FAISS existente.

### 5.4. Preparar Ollama

Se requiere Ollama accesible desde:

`http://localhost:11434`

Descargar el modelo generador:

```bash
ollama pull llama3.2:3b
```

Para repetir la evaluación mediante RAGAS, descargar también:

```bash
ollama pull qwen2.5:7b
```

En la implementación de AWS los modelos se ejecutaron mediante Ollama en un contenedor Docker.

## 6. Ejecución

### Interfaz gráfica

Desde la raíz del proyecto:

```bash
python app.py
```

Abrir la dirección local indicada por Gradio.

La interfaz permite escribir preguntas, elegir el método de recuperación y seleccionar top-k 3 o 6.

### Consultas desde terminal

Consultar documentos recuperados:

```bash
python src/rag_general.py --solo-busqueda "¿Qué dice la FDS 51 sobre el xileno?" --k 3
```

Generar una respuesta:

```bash
python src/rag_general.py "¿Qué dice la FDS 51 sobre el xileno?" --k 3
```

Comparar con top-k 6:

```bash
python src/rag_general.py "¿Qué dice la FDS 51 sobre el xileno?" --k 6
```

## 7. Verificación documental

El corpus fue contrastado con los archivos PDF originales mediante diferentes procedimientos:

- Verificación SHA-256 de los 101 documentos originales frente al inventario.
- Comprobación de lectura de los 101 PDF y disponibilidad de sus Markdown.
- Comparación automática de bloques de texto entre PDF y Markdown.
- Auditoría adicional de un bloque técnico por cada una de las 101 FDS.
- Revisión textual específica de los bloques seleccionados de las FDS 2, 14 y 80.
- Extracción de un catálogo preliminar de producto, fabricante y versión.

La auditoría técnica registró 101 bloques con similitud automática igual o superior a 0,80. Estos resultados aportan evidencia de correspondencia textual y permiten identificar documentos que requieren inspección adicional.

Evidencias principales:

- `resultados/verificacion_sha256_101.csv`
- `resultados/auditoria_pdf_markdown_101.csv`
- `resultados/auditoria_tecnica_101.csv`
- `resultados/catalogo_fds_preliminar.csv`

## 8. Evaluación mediante RAGAS

Se construyó un conjunto de 12 preguntas:

- 6 de respuesta directa.
- 2 de integración entre secciones.
- 2 de comparación.
- 2 sin respuesta disponible en el corpus.

Se generaron respuestas para top-k 3 y top-k 6, manteniendo el mismo corpus, modelo generador y procedimiento de evaluación.

Esto produjo 24 ejecuciones del sistema.

Las 10 preguntas respondibles se evaluaron mediante:

- Faithfulness: fidelidad de la respuesta al contexto recuperado.
- Context Precision: relevancia de los fragmentos recuperados.
- Context Recall: cobertura del contexto frente a la referencia.

El evaluador fue Qwen2.5 7B mediante Ollama.

Se obtuvieron 59 de las 60 mediciones previstas. La medición de Faithfulness para P06 con k=6 no finalizó debido a un tiempo de espera del evaluador y se conserva como dato no disponible.

Los archivos de preguntas, referencias, generaciones y métricas están disponibles en `resultados/`.

Esta evaluación compara top-k 3 frente a top-k 6 en la configuración híbrida; no constituye una evaluación independiente del modo exclusivamente TF-IDF.

## 9. Decisiones técnicas

Se seleccionó una arquitectura híbrida porque combina la capacidad semántica de los embeddings con la búsqueda léxica, útil para términos técnicos, nombres de sustancias, códigos y expresiones específicas de las FDS.

El uso de Ollama permite ejecutar la generación y evaluación con modelos abiertos en la infraestructura configurada.

Los índices preconstruidos se conservan para facilitar la reutilización del corpus y evitar reprocesamientos innecesarios.

## 10. Alcance y mejoras futuras

El sistema está orientado a facilitar la consulta documental de información técnica sobre seguridad química.

Entre las posibles mejoras se encuentran el enriquecimiento de metadatos, la ampliación de la trazabilidad de fuentes y la incorporación de mecanismos de verificación adicionales.

Las respuestas deben contrastarse con la FDS original cuando se utilicen para decisiones de seguridad química.

## 11. Repositorio

https://github.com/Mariana1011/RAG-FDS-Seguridad-Quimica
