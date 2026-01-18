# 📚 Document Chunker - Sistema de Fragmentación para RAG

## 📖 Descripción General

`document_chunker.py` es un módulo modular y configurable para dividir documentos Markdown en fragmentos (chunks) optimizados para sistemas de **Retrieval-Augmented Generation (RAG)**.

### 🎯 Características Principales

- ✅ **Múltiples estrategias de chunking**: Fixed-size, Semantic, Hybrid
- ✅ **Tamaños configurables**: Control total sobre tamaño de chunks y overlap
- ✅ **Preservación de estructura**: Respeta tablas, bloques de código y secciones
- ✅ **Metadatos enriquecidos**: Cada chunk incluye información contextual
- ✅ **Solapamiento inteligente**: Overlap entre chunks para mantener contexto
- ✅ **Múltiples formatos de salida**: TXT, MD, JSON
- ✅ **Estadísticas detalladas**: Información sobre el proceso de chunking
- ✅ **Fácil de modificar**: Diseño modular para personalización

---

## 🚀 Inicio Rápido

### Instalación

No requiere dependencias adicionales más allá de Python 3.7+. Todos los imports son de la biblioteca estándar.

### Uso Básico

```python
from document_chunker import DocumentChunker, ChunkConfig, ChunkingStrategy

# Configuración simple
config = ChunkConfig(
    chunk_size=1000,
    chunk_overlap=200,
    strategy=ChunkingStrategy.HYBRID
)

# Crear chunker y procesar documento
chunker = DocumentChunker(config)
chunks = chunker.chunk_document("documento.md")

# Guardar resultados
chunker.save_chunks("output/chunks", format='md')
```

### Desde Línea de Comandos

```bash
# Uso por defecto (procesa documento_concatenado.md)
python document_chunker.py

# Especificar archivo de entrada
python document_chunker.py mi_documento.md

# Especificar archivo y directorio de salida
python document_chunker.py mi_documento.md output/mis_chunks
```

---

## ⚙️ Configuración

### Clase `ChunkConfig`

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1000 | Tamaño objetivo en caracteres |
| `chunk_overlap` | int | 200 | Solapamiento entre chunks |
| `min_chunk_size` | int | 100 | Tamaño mínimo de chunk |
| `max_chunk_size` | int | 2000 | Tamaño máximo de chunk |
| `strategy` | ChunkingStrategy | HYBRID | Estrategia de división |
| `preserve_tables` | bool | True | No dividir tablas |
| `preserve_code_blocks` | bool | True | No dividir código |
| `include_metadata` | bool | True | Incluir metadatos |

### Estrategias de Chunking

#### 1. **Fixed Size** (`ChunkingStrategy.FIXED_SIZE`)
Divide el documento en chunks de tamaño fijo con overlap.

**Ventajas:**
- Chunks de tamaño predecible
- Simple y rápido
- Bueno para documentos sin estructura clara

**Cuándo usar:**
- Documentos planos sin secciones
- Cuando se necesita uniformidad en tamaño
- Textos continuos sin estructura jerárquica

```python
config = ChunkConfig(
    chunk_size=1000,
    chunk_overlap=200,
    strategy=ChunkingStrategy.FIXED_SIZE
)
```

#### 2. **Semantic** (`ChunkingStrategy.SEMANTIC`)
Respeta la estructura del documento (páginas, secciones, párrafos).

**Ventajas:**
- Preserva coherencia semántica
- Chunks con significado completo
- Respeta límites naturales del documento

**Cuándo usar:**
- Documentos bien estructurados
- Cuando la coherencia semántica es crítica
- Documentos académicos o técnicos

```python
config = ChunkConfig(
    strategy=ChunkingStrategy.SEMANTIC,
    max_chunk_size=3000  # Permite secciones más grandes
)
```

#### 3. **Hybrid** (`ChunkingStrategy.HYBRID`) - **Recomendado**
Combina ambas estrategias: respeta estructura pero divide si es necesario.

**Ventajas:**
- Balance óptimo entre coherencia y tamaño
- Adaptativo a diferentes estructuras
- Mejor opción para la mayoría de casos

**Cuándo usar:**
- Documentos con estructura variable
- Como estrategia por defecto
- Máxima flexibilidad

```python
config = ChunkConfig(
    chunk_size=1200,
    chunk_overlap=200,
    strategy=ChunkingStrategy.HYBRID  # Default
)
```

---

## 📊 Estructura de un Chunk

### Clase `Chunk`

Cada fragmento contiene:

```python
Chunk(
    content="Contenido del fragmento...",
    chunk_id=0,
    metadata={
        'page': 1,                    # Página de origen
        'section_title': 'Introduction',  # Título de sección
        'strategy': 'hybrid_semantic',    # Estrategia usada
        'source_file': 'documento.md',   # Archivo origen
        'total_chunks': 50,              # Total de chunks
        'start_pos': 0,                  # Posición inicial
        'end_pos': 1200                  # Posición final
    }
)
```

### Metadatos Disponibles

| Campo | Descripción | Disponibilidad |
|-------|-------------|----------------|
| `page` | Número de página | Si hay marcadores `## Página N` |
| `section_title` | Título de la sección | Estrategias semantic/hybrid |
| `strategy` | Estrategia de chunking usada | Todos |
| `source_file` | Nombre del archivo origen | Todos |
| `total_chunks` | Total de chunks generados | Todos |
| `start_pos` | Posición de inicio en caracteres | Fixed size |
| `end_pos` | Posición final en caracteres | Fixed size |

---

## 💾 Formatos de Salida

### 1. Archivos Individuales (`.md` o `.txt`)

```bash
output/chunks/
├── chunk_0000.md
├── chunk_0001.md
├── chunk_0002.md
└── ...
```

Cada archivo contiene:
```markdown
<!-- Chunk ID: 0 -->
<!-- Metadata: {'page': 1, 'section_title': 'Introduction', ...} -->

## 1 Introduction

The U.S. Department of Energy's (DOE's) National Renewable...
```

### 2. Formato JSON (`.json`)

```bash
output/chunks_json/
├── chunk_0000.json
├── chunk_0001.json
└── ...
```

Estructura:
```json
{
  "chunk_id": 0,
  "content": "Contenido completo del chunk...",
  "metadata": {
    "page": 1,
    "section_title": "Introduction",
    "strategy": "hybrid_semantic",
    "source_file": "documento.md",
    "total_chunks": 50
  },
  "length": 1234
}
```

### 3. Archivo Combinado

Un único archivo con todos los chunks separados:

```markdown
# Documento Dividido en Chunks para RAG

Total de chunks: 50
Configuración: ChunkConfig(...)

================================================================================

## CHUNK 0

**Metadatos:** {'page': 1, ...}
**Longitud:** 1234 caracteres

---

Contenido del chunk...

================================================================================
```

---

## 📈 Estadísticas y Monitoreo

El módulo genera automáticamente estadísticas detalladas:

```
============================================================
📊 ESTADÍSTICAS DE CHUNKING
============================================================
Total de chunks: 45
Longitud promedio: 1185 caracteres
Longitud mínima: 456 caracteres
Longitud máxima: 2340 caracteres
Longitud total: 53,325 caracteres

Distribución por estrategia:
  - hybrid_semantic: 30 chunks (66.7%)
  - hybrid_split: 15 chunks (33.3%)
============================================================
```

---

## 🔧 Casos de Uso Avanzados

### Ejemplo 1: Documentos Técnicos Largos

```python
from document_chunker import DocumentChunker, ChunkConfig, ChunkingStrategy

# Configuración para documentos técnicos
config = ChunkConfig(
    chunk_size=1500,        # Chunks más grandes para contexto
    chunk_overlap=300,      # Overlap generoso
    max_chunk_size=3000,    # Permitir secciones completas
    strategy=ChunkingStrategy.HYBRID,
    preserve_tables=True    # Crucial para docs técnicos
)

chunker = DocumentChunker(config)
chunks = chunker.chunk_document("manual_tecnico.md")

# Guardar en múltiples formatos
chunker.save_chunks("output/chunks", format='md')
chunker.save_chunks("output/chunks_json", format='json')
```

### Ejemplo 2: Documentos Cortos con Alta Coherencia

```python
config = ChunkConfig(
    chunk_size=800,         # Chunks más pequeños
    chunk_overlap=150,
    strategy=ChunkingStrategy.SEMANTIC,  # Priorizar coherencia
    min_chunk_size=200
)

chunker = DocumentChunker(config)
chunks = chunker.chunk_document("articulo.md")
```

### Ejemplo 3: Procesamiento en Batch

```python
import os
from document_chunker import create_chunks_from_file

# Procesar múltiples documentos
documentos = [
    "doc1.md",
    "doc2.md",
    "doc3.md"
]

for doc in documentos:
    output_dir = f"output/chunks_{os.path.splitext(doc)[0]}"
    chunks = create_chunks_from_file(
        input_file=doc,
        output_dir=output_dir,
        chunk_size=1200,
        overlap=200,
        strategy='hybrid'
    )
    print(f"✓ {doc}: {len(chunks)} chunks generados")
```

### Ejemplo 4: Análisis de Chunks

```python
# Crear chunks
chunker = DocumentChunker()
chunks = chunker.chunk_document("documento.md")

# Analizar chunks
for chunk in chunks:
    print(f"Chunk {chunk.chunk_id}:")
    print(f"  - Longitud: {len(chunk.content)}")
    print(f"  - Página: {chunk.metadata.get('page', 'N/A')}")
    print(f"  - Sección: {chunk.metadata.get('section_title', 'N/A')}")
    print()

# Filtrar chunks por página
chunks_pagina_1 = [c for c in chunks if c.metadata.get('page') == 1]
print(f"Chunks de página 1: {len(chunks_pagina_1)}")
```

---

## 🎨 Personalización

### Añadir Nueva Estrategia

```python
def _chunk_custom(self, content: str) -> List[Chunk]:
    """Estrategia personalizada de chunking"""
    chunks = []
    # Tu lógica aquí
    return chunks

# Registrar en el método chunk_document
if self.config.strategy == ChunkingStrategy.CUSTOM:
    self.chunks = self._chunk_custom(content)
```

### Modificar Detección de Tablas

```python
def _is_table(self, text: str) -> bool:
    """Lógica personalizada para detectar tablas"""
    # Tu lógica mejorada aquí
    return False
```

### Añadir Metadatos Personalizados

```python
# En el método chunk_document
for chunk in self.chunks:
    chunk.metadata['custom_field'] = "valor personalizado"
    chunk.metadata['word_count'] = len(chunk.content.split())
    chunk.metadata['has_equations'] = '$' in chunk.content
```

---

## 📋 Recomendaciones

### Tamaños de Chunk Sugeridos

| Caso de Uso | chunk_size | overlap | Estrategia |
|-------------|-----------|---------|------------|
| **RAG Conversacional** | 800-1000 | 150-200 | HYBRID |
| **Búsqueda Semántica** | 1200-1500 | 200-300 | SEMANTIC |
| **Análisis de Documentos** | 1500-2000 | 300-400 | HYBRID |
| **Q&A Específico** | 600-800 | 100-150 | SEMANTIC |
| **Embeddings** | 1000-1200 | 200 | HYBRID |

### Mejores Prácticas

1. **Overlap Adecuado**: Usa 15-20% del chunk_size como overlap
2. **Preservar Tablas**: Siempre mantén `preserve_tables=True` para documentos técnicos
3. **Estrategia HYBRID**: Es la más versátil para la mayoría de casos
4. **Validar Resultados**: Revisa las estadísticas y algunos chunks de ejemplo
5. **Metadatos**: Aprovecha los metadatos para filtrado y contexto en RAG

### Ajustar para tu Modelo de Embeddings

```python
# Para modelos con límite de tokens (ej: 512 tokens)
# Aproximadamente 1 token ≈ 4 caracteres en inglés

MAX_TOKENS = 512
CHARS_PER_TOKEN = 4

config = ChunkConfig(
    chunk_size=MAX_TOKENS * CHARS_PER_TOKEN * 0.8,  # ~1600 chars
    chunk_overlap=200
)
```

---

## 🔍 Solución de Problemas

### Chunks Demasiado Pequeños

**Síntoma**: Muchos chunks de tamaño menor al esperado

**Solución**:
```python
config = ChunkConfig(
    min_chunk_size=500,      # Aumentar mínimo
    max_chunk_size=3000,     # Aumentar máximo
    strategy=ChunkingStrategy.SEMANTIC  # Preferir secciones completas
)
```

### Chunks Demasiado Grandes

**Síntoma**: Algunos chunks exceden el tamaño máximo

**Solución**:
```python
config = ChunkConfig(
    max_chunk_size=1500,     # Reducir máximo
    strategy=ChunkingStrategy.FIXED_SIZE  # Forzar tamaño fijo
)
```

### Tablas Divididas

**Síntoma**: Tablas cortadas en múltiples chunks

**Solución**:
```python
config = ChunkConfig(
    preserve_tables=True,
    max_chunk_size=5000  # Permitir chunks grandes para tablas
)
```

### Pérdida de Contexto

**Síntoma**: Chunks sin suficiente contexto

**Solución**:
```python
config = ChunkConfig(
    chunk_overlap=400,       # Aumentar overlap
    chunk_size=1500          # Chunks más grandes
)
```

---

## 🚦 Integración con RAG

### Pipeline Completo RAG

```python
from document_chunker import DocumentChunker, ChunkConfig

# 1. Chunking
config = ChunkConfig(chunk_size=1200, chunk_overlap=200)
chunker = DocumentChunker(config)
chunks = chunker.chunk_document("documento.md")

# 2. Generar Embeddings (ejemplo con OpenAI)
import openai

embeddings = []
for chunk in chunks:
    response = openai.Embedding.create(
        input=chunk.content,
        model="text-embedding-ada-002"
    )
    embeddings.append({
        'chunk_id': chunk.chunk_id,
        'embedding': response['data'][0]['embedding'],
        'metadata': chunk.metadata
    })

# 3. Almacenar en Vector DB
# (Ejemplo conceptual - ajustar según tu DB)
vector_db.insert(embeddings)
```

---

## 📚 Recursos Adicionales

### Documentos Relacionados
- [parse_local.py](parse_local_readme.md) - Parsing de PDFs
- [README_MODULE.md](README_MODULE.md) - Descripción general del proyecto

### Referencias sobre RAG
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chunking Strategies for RAG](https://www.pinecone.io/learn/chunking-strategies/)

---

## 📝 Changelog

### v1.0 (Enero 2026)
- ✨ Implementación inicial
- ✨ Tres estrategias de chunking
- ✨ Soporte para metadatos
- ✨ Múltiples formatos de salida
- ✨ Estadísticas detalladas

---

## 🤝 Contribuciones

Para modificar o extender el módulo:

1. Las estrategias de chunking están en métodos `_chunk_*`
2. La detección de estructura está en `_extract_sections`
3. El manejo de tablas está en `_is_table`
4. Los metadatos se añaden en `chunk_document`

---

## 📄 Licencia

Parte del proyecto de Sistema RAG - Enero 2026

---

**¿Preguntas o sugerencias?** El código está diseñado para ser modificado fácilmente. Revisa los comentarios en el código para más detalles.
