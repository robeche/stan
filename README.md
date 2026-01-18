# 🤖 Sistema RAG Modular

**Sistema completo de Retrieval-Augmented Generation (RAG)** para procesar, indexar y consultar documentos técnicos de forma inteligente.

## ¿Qué hace este sistema?

Este proyecto convierte documentos complejos (PDFs, DOCX, etc.) en un sistema de búsqueda semántica inteligente. Permite hacer preguntas en lenguaje natural y obtener respuestas precisas basadas en el contenido de los documentos, indicando siempre las fuentes.

**Flujo completo:**
1. 📄 **Parsea** documentos PDF → extrae texto, tablas e imágenes
2. ✂️ **Divide** el contenido en fragmentos semánticos (chunks)
3. 🧠 **Genera embeddings** (representaciones vectoriales) de cada fragmento
4. 🗄️ **Indexa** en una base de datos vectorial (ChromaDB)
5. 🔍 **Busca** fragmentos relevantes para cualquier consulta
6. 🎯 **Reordena** resultados por relevancia (reranking)

## 🚀 Dos Formas de Usar el Sistema

### **Opción 1: Aplicación Web Django** 🌐

Interfaz gráfica completa con administración de documentos y chatbot.

**Características:**
- Panel de administración para subir y procesar documentos
- Chatbot público para hacer preguntas sobre los documentos
- Procesamiento automático en segundo plano
- Visualización de fragmentos extraídos, imágenes y tablas
- Dashboard con estadísticas

**Ideal para:** Uso en producción, múltiples usuarios, interfaz amigable

👉 **[Ver guía completa de la WebApp](WebApp/README.md)**

```bash
cd WebApp
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

### **Opción 2: Pipeline Manual con Python** 🐍

Usa los módulos directamente en tu código Python para máximo control.

**Ideal para:** Integración personalizada, scripting, notebooks, experimentación

```python
# Pipeline completo en pocas líneas
from parse_local import NemotronParser
from document_chunker import DocumentChunker
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from reranker import Reranker

# 1. Parsear documento
parser = NemotronParser()
parser.process_pdf("documento.pdf", "output/doc")

# 2. Dividir en chunks
chunker = DocumentChunker(chunk_size=2000, overlap=200)
chunks = chunker.chunk_document("output/doc/documento_concatenado.md")

# 3. Generar embeddings
generator = EmbeddingGenerator("bge-m3")
generator.process_chunks_directory("chunks/", "embeddings/")

# 4. Indexar en ChromaDB
store = VectorStore(persist_directory="chroma_db")
store.add_embeddings_from_directory("embeddings/")

# 5. Buscar con reranking
query = "¿Cuáles son las especificaciones técnicas?"
query_emb = generator.generate_embedding(query)
results = store.query_by_embedding(query_emb, n_results=20)

reranker = Reranker("bge-reranker-v2-m3")
final = reranker.rerank_results(query, results, top_k=5)

# Mostrar resultados
for i, r in enumerate(final, 1):
    print(f"{i}. {r['id']} (score: {r['rerank_score']:.4f})")
    print(f"   {r['document'][:100]}...\n")
```

👉 **Ver ejemplos completos en:** `ejemplos_*.py`

## 📚 Documentación Detallada por Módulo

Cada módulo tiene su propia documentación técnica:

1. **[README_MODULE.md](README_MODULE.md)** - Parser de documentos PDF (Nemotron)
2. **[README_CHUNKER.md](README_CHUNKER.md)** - Sistema de chunking inteligente
3. **[README_EMBEDDINGS.md](README_EMBEDDINGS.md)** - Generación de embeddings (BGE-M3)
4. **[README_VECTORSTORE.md](README_VECTORSTORE.md)** - Base de datos vectorial (ChromaDB)
5. **[README_RERANKING.md](README_RERANKING.md)** - Sistema de reranking (BGE-reranker)

## 🛠️ Instalación Rápida

### Requisitos Previos
- Python 3.11+
- GPU NVIDIA (opcional, pero recomendado para mejor rendimiento)
- 8GB+ RAM

### Configuración Básica

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU-USUARIO/TU-REPO.git
cd TU-REPO

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Configurar variables de entorno (si usas APIs externas)
cp .env.example .env
# Editar .env con tus API keys si es necesario
```

### Verificar Instalación

```bash
# Test de embeddings
python test_embeddings_install.py
```

## 🎯 Casos de Uso

### ✅ Ideal para:
- 📚 **Sistemas Q&A** sobre documentación técnica
- 🔍 **Búsqueda semántica** en corpus grandes
- 📖 **Asistentes de lectura** de manuales y especificaciones
- 🎓 **Herramientas educativas** con material extenso
- 🏢 **Knowledge bases corporativas**
- 🤝 **Chatbots especializados** en dominios específicos

### Ejemplo Real: Turbina Eólica NREL 5MW
Este proyecto incluye un caso de uso completo con documentación técnica de la turbina NREL 5MW:
- ✅ 24 fragmentos de especificaciones técnicas
- ✅ Búsquedas sobre diseño de palas, torre, capacidad
- ✅ Sistema funcionando con alta precisión
- ✅ Listo para integración con LLMs

## 🎓 Guías Rápidas

### Procesar tu Primer Documento

```bash
# Ejecutar ejemplo completo
python ejemplos_reranking.py 4

# Esto hará:
# 1. Crear chunks del documento NREL
# 2. Generar embeddings
# 3. Indexar en ChromaDB
# 4. Realizar búsquedas con reranking
```

### Usar la WebApp

```bash
cd WebApp
python manage.py runserver

# Acceder a:
# - Administración: http://localhost:8000/admin/
# - Chatbot: http://localhost:8000/chat/
```

### Integrar en tu Código

```python
# Ejemplo mínimo
from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator

# Cargar sistema existente
store = VectorStore(persist_directory="chroma_db")
generator = EmbeddingGenerator("bge-m3")

# Hacer una consulta
query = "tu pregunta aquí"
query_emb = generator.generate_embedding(query)
results = store.query_by_embedding(query_emb, n_results=5)

for r in results:
    print(f"- {r['document'][:100]}...")
```

## 🔧 Configuración Avanzada

### Modelos Disponibles

**Embeddings:**
- `bge-m3` - Recomendado: multilingüe, 1024 dims
- `bge-base` - Inglés, 768 dims, rápido
- `minilm` - Muy rápido, 384 dims

**Reranking:**
- `bge-reranker-v2-m3` - Recomendado: multilingüe
- `bge-reranker-base` - Rápido, inglés
- `ms-marco-small` - Muy rápido

### Parámetros Recomendados

```python
# Para documentos técnicos largos
DocumentChunker(
    chunk_size=2000,              # Fragmentos medianos
    overlap=200,                  # 10% de overlap
    strategy="hybrid_semantic"    # Mejor calidad
)

# Para documentos cortos o preguntas específicas
DocumentChunker(
    chunk_size=800,
    overlap=100,
    strategy="semantic"
)
```


## 📖 Recursos y Referencias

### Papers Técnicos
- [BGE Embeddings](https://arxiv.org/abs/2309.07597) - Base de los modelos de embeddings
- [RAG](https://arxiv.org/abs/2005.11401) - Fundamentos de Retrieval-Augmented Generation

### Herramientas
- [ChromaDB](https://docs.trychroma.com/) - Base de datos vectorial
- [Sentence Transformers](https://www.sbert.net/) - Framework de embeddings


## 📝 Changelog

### v1.0.0 (Enero 2026)
- ✅ Sistema RAG completo con 5 módulos
- ✅ Aplicación web Django con admin panel y chatbot
- ✅ Procesamiento automático en background (Celery)
- ✅ Documentación completa de todos los componentes
- ✅ Scripts de ejemplo para cada módulo
- ✅ Configuración de seguridad (gitignore, variables de entorno)

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y de investigación.

---

