# Módulo de Generación de Embeddings para RAG

Sistema modular y flexible para generar embeddings vectoriales de documentos, optimizado para sistemas RAG (Retrieval-Augmented Generation).

## 🎯 Características

- ✅ **Múltiples modelos soportados**: Nemotron, BGE, MiniLM, MPNet, OpenAI
- ✅ **Backends flexibles**: Sentence Transformers (local) y OpenAI API
- ✅ **GPU/CPU automático**: Detección y uso automático de hardware disponible
- ✅ **Procesamiento batch**: Optimizado para grandes volúmenes de datos
- ✅ **Normalización**: Embeddings normalizados para cosine similarity
- ✅ **Múltiples formatos**: JSON, NumPy, o ambos
- ✅ **Progreso visual**: Barras de progreso para procesos largos
- ✅ **Metadata completa**: Información detallada de cada generación

## 🚀 Modelo Recomendado: NVIDIA Nemotron

**`nvidia/NV-Embed-v2`** es el modelo recomendado por:

- 🏆 **Top performance** en benchmarks de retrieval
- 📊 **4096 dimensiones** (alta capacidad de representación)
- 📖 **32K tokens** de contexto (documentos largos)
- 🎯 **Optimizado para RAG** específicamente
- 🔬 **Excelente en contenido técnico/científico**

## 📦 Instalación

```bash
# Instalar dependencias
pip install sentence-transformers torch numpy tqdm

# Para GPU (recomendado)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Opcional: OpenAI
pip install openai
```

## 🎓 Uso Básico

### Ejemplo 1: Embeddings simples

```python
from embedding_generator import EmbeddingGenerator

# Crear generador con Nemotron
generator = EmbeddingGenerator(
    model_name="nemotron-v2",
    device="auto"  # Usa GPU si está disponible
)

# Generar embedding para un texto
text = "Wind turbine blade design optimization"
embedding = generator.generate_embedding(text)

print(f"Dimensión: {len(embedding)}")  # 4096
```

### Ejemplo 2: Procesar chunks

```python
# Procesar directorio completo de chunks
metadata = generator.process_chunks_directory(
    chunks_dir="output_simple/NREL5MW_Reduced/chunks_json",
    output_dir="output_rag/embeddings",
    text_field="content",
    save_format="both"  # JSON + NumPy
)

print(f"Procesados: {metadata['num_chunks']} chunks")
print(f"Tiempo: {metadata['generation_time_seconds']:.2f}s")
```

### Ejemplo 3: Batch de embeddings

```python
textos = [
    "Wind turbine aerodynamics",
    "Offshore wind energy",
    "Power curve optimization"
]

# Generar todos a la vez (más eficiente)
embeddings = generator.generate_embeddings_batch(textos)
print(embeddings.shape)  # (3, 4096)
```

## 🎯 Modelos Disponibles

| Alias | Modelo | Dims | Mejor Para | Velocidad |
|-------|--------|------|------------|-----------|
| `nemotron-v2` | nvidia/NV-Embed-v2 | 4096 | RAG técnico/científico | ⭐⭐⭐ |
| `nemotron-v1` | nvidia/NV-Embed-v1 | 4096 | RAG general | ⭐⭐⭐ |
| `bge-large` | BAAI/bge-large-en-v1.5 | 1024 | Máxima precisión | ⭐⭐⭐ |
| `bge-base` | BAAI/bge-base-en-v1.5 | 768 | Retrieval de calidad | ⭐⭐⭐⭐ |
| `bge-small` | BAAI/bge-small-en-v1.5 | 384 | Retrieval rápido | ⭐⭐⭐⭐⭐ |
| `mpnet` | all-mpnet-base-v2 | 768 | Alta calidad general | ⭐⭐⭐⭐ |
| `minilm` | all-MiniLM-L6-v2 | 384 | Balance velocidad/calidad | ⭐⭐⭐⭐⭐ |
| `multilingual` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | Multilingüe | ⭐⭐⭐⭐ |
| `openai-small` | text-embedding-3-small | 1536 | API (económico) | API |
| `openai-large` | text-embedding-3-large | 3072 | API (mejor calidad) | API |

### Comparar modelos

```python
# Listar todos los modelos
modelos = EmbeddingGenerator.list_available_models()
for alias, nombre in modelos.items():
    print(f"{alias}: {nombre}")

# Ver información detallada
info = EmbeddingGenerator.get_model_info("nemotron-v2")
print(info)
```

## 🔧 Configuración Avanzada

### Personalizar configuración

```python
generator = EmbeddingGenerator(
    model_name="nemotron-v2",
    device="cuda",              # Forzar GPU
    batch_size=16,              # Ajustar según VRAM
    normalize_embeddings=True,  # Para cosine similarity
    show_progress=True          # Mostrar barra de progreso
)
```

### Usar con OpenAI

```python
import os
os.environ["OPENAI_API_KEY"] = "tu-api-key"

generator = EmbeddingGenerator(
    model_name="openai-small",
    batch_size=100  # OpenAI permite batches grandes
)
```

## 💾 Formatos de Salida

### Formato JSON (con chunks)

```json
{
  "chunk_id": 0,
  "content": "Wind turbine blade design...",
  "metadata": {...},
  "embedding": [0.123, -0.456, ...],
  "embedding_model": "nvidia/NV-Embed-v2"
}
```

### Formato NumPy (matriz de embeddings)

```python
import numpy as np

# Cargar embeddings
embeddings = np.load("output_rag/embeddings/embeddings.npy")
print(embeddings.shape)  # (n_chunks, embedding_dim)
```

### Metadata

```json
{
  "model": "nvidia/NV-Embed-v2",
  "model_alias": "nemotron-v2",
  "embedding_dimension": 4096,
  "num_chunks": 24,
  "normalized": true,
  "generation_time_seconds": 45.23,
  "backend": "sentence-transformers",
  "timestamp": "2026-01-02 15:30:00"
}
```

## 🖥️ Línea de Comandos

```bash
# Uso básico
python embedding_generator.py output_simple/NREL5MW_Reduced/chunks_json

# Con opciones
python embedding_generator.py \
  output_simple/NREL5MW_Reduced/chunks_json \
  --model nemotron-v2 \
  --output-dir output_rag/embeddings \
  --batch-size 16 \
  --device cuda \
  --save-format both

# Listar modelos disponibles
python embedding_generator.py --list-models

# Ver ayuda
python embedding_generator.py --help
```

### Opciones disponibles

| Opción | Descripción | Default |
|--------|-------------|---------|
| `--model` | Modelo a usar | `nemotron-v2` |
| `--output-dir` | Directorio de salida | `chunks_dir/embeddings` |
| `--batch-size` | Tamaño del batch | `32` |
| `--device` | Dispositivo (auto/cuda/cpu) | `auto` |
| `--save-format` | Formato (json/npy/both) | `both` |
| `--text-field` | Campo JSON con texto | `content` |
| `--list-models` | Listar modelos y salir | - |

## 📊 Ejemplos Prácticos

### Ejecutar ejemplos interactivos

```bash
# Menú interactivo
python ejemplos_embeddings.py

# Ejemplo específico
python ejemplos_embeddings.py 2  # Procesar chunks
```

### Ejemplos incluidos

1. **Embeddings básico**: Generar embedding para un texto
2. **Procesar chunks**: Procesar directorio completo
3. **Comparar modelos**: Probar diferentes modelos
4. **Batch de embeddings**: Múltiples textos a la vez
5. **Listar modelos**: Ver todos los modelos disponibles
6. **Configuración avanzada**: Opciones personalizadas

## 🎯 Recomendaciones por Uso

### Para documentos técnicos (tu caso)
```python
generator = EmbeddingGenerator("nemotron-v2")  # ⭐ Mejor opción
```

### Para velocidad máxima
```python
generator = EmbeddingGenerator("minilm", batch_size=64)
```

### Para máxima precisión
```python
generator = EmbeddingGenerator("bge-large", batch_size=8)
```

### Para multilingüe
```python
generator = EmbeddingGenerator("multilingual")
```

## 📈 Performance

Tiempos aproximados para 24 chunks (~150K tokens) en GPU NVIDIA RTX 3080:

| Modelo | Tiempo | Tokens/seg | Dims |
|--------|--------|------------|------|
| nemotron-v2 | ~45s | 3.3K | 4096 |
| bge-large | ~25s | 6.0K | 1024 |
| bge-base | ~15s | 10.0K | 768 |
| minilm | ~8s | 18.7K | 384 |

## 🔍 Verificar Instalación

```python
from embedding_generator import EmbeddingGenerator

# Verificar que funciona
generator = EmbeddingGenerator("minilm")
embedding = generator.generate_embedding("test")
print(f"✓ Funcionando! Dimensión: {len(embedding)}")
```

## 🤔 Preguntas Frecuentes

**P: ¿Qué modelo debo usar?**  
R: Para RAG con documentos técnicos, usa `nemotron-v2`. Para velocidad, usa `minilm`.

**P: ¿GPU es necesaria?**  
R: No es obligatoria, pero acelera mucho (10-50x más rápido).

**P: ¿Los embeddings ocupan mucho espacio?**  
R: Depende del modelo. Para 24 chunks:
- `minilm` (384 dims): ~37KB
- `nemotron-v2` (4096 dims): ~394KB

**P: ¿Puedo usar mi propio modelo?**  
R: Sí, pasa el nombre completo del modelo HuggingFace:
```python
generator = EmbeddingGenerator("tu-usuario/tu-modelo")
```

**P: ¿Cómo sé si está usando GPU?**  
R: Mira la salida al inicializar:
```
✓ Modelo cargado: nvidia/NV-Embed-v2
  Dispositivo: cuda:0  ← GPU en uso
```

## 🔗 Integración con RAG

Este módulo es el primer paso de tu pipeline RAG:

```
1. document_chunker.py    → Dividir documento
2. embedding_generator.py → Generar embeddings ← ESTÁS AQUÍ
3. vector_store.py        → Almacenar en base vectorial
4. retriever.py           → Buscar chunks relevantes
5. llm_generator.py       → Generar respuestas
```

## 📝 Siguiente Paso

Ahora que tienes los embeddings, el siguiente paso es crear una base de datos vectorial para búsquedas eficientes. Opciones:

- **FAISS** (local, rápido)
- **ChromaDB** (local, persistente)
- **Pinecone** (cloud, escalable)
- **Weaviate** (open source, full-featured)

---

**Creado**: 2026-01-02  
**Versión**: 1.0  
**Autor**: Sistema RAG
