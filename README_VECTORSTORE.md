# Módulo Vector Store con ChromaDB

Sistema de base de datos vectorial para almacenar y buscar embeddings de documentos usando ChromaDB.

## 🎯 Características

- ✅ **Almacenamiento persistente** de embeddings
- ✅ **Búsqueda por similitud** (cosine, L2, inner product)
- ✅ **Filtrado por metadata** avanzado
- ✅ **Operaciones batch** eficientes
- ✅ **Múltiples colecciones** en una BD
- ✅ **Integración directa** con embedding_generator.py
- ✅ **Fácil de usar** con API intuitiva

## 📦 Instalación

```bash
pip install chromadb numpy tqdm
```

## 🚀 Uso Básico

### Ejemplo 1: Crear y cargar embeddings

```python
from vector_store import VectorStore

# Crear vector store
store = VectorStore(
    persist_directory="output_rag/chroma_db",
    collection_name="my_documents"
)

# Cargar embeddings desde directorio de chunks JSON
stats = store.add_embeddings_from_directory(
    embeddings_dir="output_rag/embeddings"
)

print(f"Cargados {stats['chunks_loaded']} chunks")
```

### Ejemplo 2: Buscar por similitud

```python
from embedding_generator import EmbeddingGenerator

# Generar embedding para query
generator = EmbeddingGenerator("bge-m3")
query_embedding = generator.generate_embedding("What is the blade design?")

# Buscar documentos similares
results = store.query_by_embedding(
    query_embedding=query_embedding,
    n_results=5
)

# Mostrar resultados
for result in results['results']:
    print(f"{result['id']}: {result['similarity']:.4f}")
    print(f"{result['document'][:100]}...")
```

### Ejemplo 3: Filtrar por metadata

```python
# Buscar solo chunks largos
results = store.query_by_embedding(
    query_embedding=query_embedding,
    n_results=5,
    where={"length": {"$gt": 500}}
)

# Buscar documentos que contengan palabra específica
results = store.query_by_embedding(
    query_embedding=query_embedding,
    n_results=5,
    where_document={"$contains": "turbine"}
)
```

## 🔧 API Completa

### Inicialización

```python
store = VectorStore(
    persist_directory="output_rag/chroma_db",  # Directorio persistente
    collection_name="documents",                # Nombre de colección
    distance_metric="cosine"                    # 'cosine', 'l2', 'ip'
)
```

### Cargar embeddings

```python
stats = store.add_embeddings_from_directory(
    embeddings_dir="path/to/embeddings",
    show_progress=True
)
# Retorna: {'chunks_loaded': 24, 'total_documents': 24, ...}
```

### Buscar por embedding

```python
results = store.query_by_embedding(
    query_embedding=[0.1, 0.2, ...],  # Vector de embedding
    n_results=5,                       # Número de resultados
    where={"chunk_id": {"$gt": 10}},  # Filtros de metadata (opcional)
    where_document={"$contains": "x"} # Filtros de contenido (opcional)
)
```

**Formato de resultados:**
```python
{
    'query_embedding_dim': 1024,
    'n_results': 5,
    'results': [
        {
            'id': 'chunk_0005',
            'document': 'texto del documento...',
            'metadata': {'chunk_id': 5, 'length': 1024, ...},
            'distance': 0.35,
            'similarity': 0.65
        },
        ...
    ]
}
```

### Obtener por IDs

```python
docs = store.get_by_ids(['chunk_0001', 'chunk_0005'])
# Retorna: {'ids': [...], 'documents': [...], 'metadatas': [...], 'embeddings': [...]}
```

### Eliminar documentos

```python
store.delete_by_ids(['chunk_0001', 'chunk_0002'])
```

### Actualizar metadata

```python
store.update_metadata('chunk_0001', {'status': 'reviewed'})
```

### Estadísticas

```python
stats = store.get_stats()
# {'collection_name': 'documents', 'total_documents': 24, ...}

# Ver muestra
sample = store.peek(limit=3)

# Listar colecciones
collections = store.list_collections()
```

### Resetear colección

```python
store.reset_collection()  # ⚠️ Elimina todos los documentos
```

## 🔍 Operadores de Filtrado

### Metadata (where)

```python
# Igualdad
where={"chunk_id": 5}

# Mayor que
where={"length": {"$gt": 500}}

# Menor que  
where={"length": {"$lt": 1000}}

# Mayor o igual
where={"chunk_id": {"$gte": 10}}

# Menor o igual
where={"chunk_id": {"$lte": 20}}

# No igual
where={"chunk_id": {"$ne": 5}}

# En lista
where={"chunk_id": {"$in": [1, 2, 3]}}

# No en lista
where={"chunk_id": {"$nin": [4, 5, 6]}}

# AND lógico
where={"$and": [
    {"length": {"$gt": 500}},
    {"chunk_id": {"$lt": 10}}
]}

# OR lógico
where={"$or": [
    {"length": {"$gt": 1000}},
    {"chunk_id": {"$lt": 5}}
]}
```

### Documentos (where_document)

```python
# Contiene texto
where_document={"$contains": "turbine"}

# No contiene texto
where_document={"$not_contains": "error"}

# AND/OR también disponibles
where_document={"$and": [
    {"$contains": "blade"},
    {"$contains": "design"}
]}
```

## 📊 Métricas de Distancia

### Cosine (recomendada para embeddings normalizados)
```python
store = VectorStore(distance_metric="cosine")
# Rango: 0 (idéntico) a 2 (opuesto)
# Similarity = 1 - distance
```

### L2 (distancia euclidiana)
```python
store = VectorStore(distance_metric="l2")
# Rango: 0 (idéntico) a ∞
```

### Inner Product
```python
store = VectorStore(distance_metric="ip")
# Útil para embeddings no normalizados
```

## 🖥️ Línea de Comandos

```bash
# Cargar embeddings
python vector_store.py load output_rag/embeddings \
  --persist-dir output_rag/chroma_db \
  --collection documents

# Ver estadísticas
python vector_store.py stats \
  --persist-dir output_rag/chroma_db \
  --collection documents

# Buscar (requiere archivo JSON con embedding)
python vector_store.py query query_embedding.json \
  --persist-dir output_rag/chroma_db \
  --n-results 5

# Resetear colección
python vector_store.py reset \
  --persist-dir output_rag/chroma_db \
  --collection documents
```

## 📝 Ejemplos Avanzados

### Pipeline completo de búsqueda

```python
from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator

# Setup
store = VectorStore(persist_directory="output_rag/chroma_db")
generator = EmbeddingGenerator("bge-m3")

# Query
query = "What are the turbine specifications?"
query_emb = generator.generate_embedding(query)

# Buscar con filtros
results = store.query_by_embedding(
    query_embedding=query_emb,
    n_results=10,
    where={"length": {"$gt": 300}},
    where_document={"$contains": "specification"}
)

# Procesar resultados
for r in results['results']:
    print(f"Similarity: {r['similarity']:.4f}")
    print(f"Content: {r['document'][:200]}...")
```

### Múltiples colecciones

```python
# Colección para documentos técnicos
tech_store = VectorStore(
    persist_directory="output_rag/chroma_db",
    collection_name="technical_docs"
)

# Colección para manuales
manual_store = VectorStore(
    persist_directory="output_rag/chroma_db",
    collection_name="manuals"
)

# Buscar en ambas
tech_results = tech_store.query_by_embedding(query_emb, n_results=3)
manual_results = manual_store.query_by_embedding(query_emb, n_results=3)

# Combinar resultados
all_results = tech_results['results'] + manual_results['results']
all_results.sort(key=lambda x: x['similarity'], reverse=True)
```

### Búsqueda híbrida (semántica + keyword)

```python
# 1. Búsqueda semántica amplia
semantic_results = store.query_by_embedding(
    query_embedding=query_emb,
    n_results=20
)

# 2. Filtrar por keywords
filtered = [
    r for r in semantic_results['results']
    if 'blade' in r['document'].lower() and 'design' in r['document'].lower()
]

# 3. Top K final
top_results = filtered[:5]
```

## 🔄 Integración con Reranking

```python
from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator
from reranker import Reranker

# Setup
store = VectorStore(persist_directory="output_rag/chroma_db")
generator = EmbeddingGenerator("bge-m3")
reranker = Reranker("bge-reranker-v2-m3")

# Pipeline
query = "blade specifications"
query_emb = generator.generate_embedding(query)

# 1. Recuperar candidatos (más que lo necesario)
results = store.query_by_embedding(query_emb, n_results=20)

# 2. Reranking para obtener top 5
reranked = reranker.rerank_results(
    query=query,
    search_results=results,
    top_k=5
)

# 3. Usar resultados refinados
for r in reranked['results']:
    print(f"Score: {r['rerank_score']:.4f}")
    print(r['document'][:100])
```

## 🎯 Mejores Prácticas

### Performance

```python
# ✅ Cargar embeddings en batch (más rápido)
store.add_embeddings_from_directory("embeddings/")

# ❌ Evitar agregar uno por uno
for embedding in embeddings:
    store.add(...)  # Lento
```

### Filtros

```python
# ✅ Usar filtros específicos para reducir búsqueda
where={"length": {"$gt": 500}, "chunk_id": {"$lt": 20}}

# ❌ Filtrar después de recuperar todo
results = store.query(...)  
filtered = [r for r in results if r['metadata']['length'] > 500]  # Ineficiente
```

### Métricas

```python
# ✅ Cosine para embeddings normalizados (BGE, OpenAI)
store = VectorStore(distance_metric="cosine")

# ✅ L2 para embeddings no normalizados
store = VectorStore(distance_metric="l2")
```

## 🐛 Troubleshooting

**P: Error "failed to extract enum MetadataValue"**  
R: ChromaDB no acepta valores `None` en metadata. El módulo filtra automáticamente estos valores.

**P: Búsquedas lentas**  
R: 
- Usa filtros `where` para reducir espacio de búsqueda
- Considera indexar metadata frecuentes
- Reduce `n_results` si solo necesitas pocos

**P: "Collection not found"**  
R: La colección se crea automáticamente. Verifica el nombre con `list_collections()`

**P: Documentos duplicados**  
R: ChromaDB usa IDs únicos. Si insertas con mismo ID, sobrescribe.

## 📊 Benchmarks

Para 24 chunks (1024 dims, BGE-M3):
- **Carga inicial**: ~0.5s
- **Query (top 10)**: ~5-10ms
- **Tamaño en disco**: ~500KB (incluyendo índices)
- **Memoria**: ~2-3MB en uso

## 🔗 Integración con Pipeline RAG

```python
# 1. Chunking
from document_chunker import DocumentChunker
chunker = DocumentChunker()
chunks = chunker.chunk_document("doc.md", output_dir="chunks/")

# 2. Embeddings
from embedding_generator import EmbeddingGenerator
generator = EmbeddingGenerator("bge-m3")
generator.process_chunks_directory("chunks/", "embeddings/")

# 3. Vector Store ← ESTÁS AQUÍ
from vector_store import VectorStore
store = VectorStore()
store.add_embeddings_from_directory("embeddings/")

# 4. Retrieval + Reranking
from reranker import Reranker
reranker = Reranker("bge-reranker-v2-m3")

query = "Your question here"
query_emb = generator.generate_embedding(query)
results = store.query_by_embedding(query_emb, n_results=20)
final = reranker.rerank_results(query, results, top_k=5)

# 5. LLM (próximo paso)
# context = build_context(final)
# response = llm.generate(query, context)
```

## 📚 Recursos

- **Documentación ChromaDB**: https://docs.trychroma.com/
- **Paper BGE**: https://arxiv.org/abs/2309.07597
- **Guía RAG**: https://www.pinecone.io/learn/retrieval-augmented-generation/

---

**Creado**: 2026-01-02  
**Versión**: 1.0  
**Autor**: Sistema RAG
