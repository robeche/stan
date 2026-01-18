# 🎯 Vista del Chatbot RAG - Proyecto Completo

## 📊 Resumen del Sistema

El proyecto consta de **dos módulos principales** integrados en una aplicación Django:

### 🔐 Módulo 1: Panel de Administración
**URL:** `/admin-panel/`  
**Acceso:** Solo usuarios administradores

**Funcionalidades:**
- Subir documentos (PDF, DOCX, DOC, TXT, MD)
- Procesamiento automático en background con Celery
- 4 etapas: Parsing → Chunking → Embeddings → Indexing
- Visualización de progreso en tiempo real
- Explorar páginas anotadas con bounding boxes (PNG)
- Ver imágenes y tablas extraídas
- Dashboard con estadísticas
- Logs detallados del procesamiento
- Función de reprocesamiento

### 💬 Módulo 2: Chatbot Público
**URL:** `/` (raíz del sitio)  
**Acceso:** Público (todos los usuarios)

**Funcionalidades:**
- Interfaz de chat moderna y responsive
- Búsqueda vectorial en ChromaDB
- Respuestas basadas en documentos indexados
- Muestra fuentes y referencias
- Historial de conversación
- Sesiones independientes

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        USUARIO                                  │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
     ┌───────▼────────┐                  ┌────────▼─────────┐
     │  ADMIN PANEL   │                  │    CHATBOT       │
     │  (Privado)     │                  │   (Público)      │
     └───────┬────────┘                  └────────┬─────────┘
             │                                    │
             │ 1. Upload Document                 │ 1. User Query
             ▼                                    ▼
     ┌───────────────────┐              ┌─────────────────────┐
     │   CELERY TASK     │              │  EMBEDDING GEN      │
     │   (Background)    │              │   (BGE-M3)          │
     └───────┬───────────┘              └─────────┬───────────┘
             │                                    │
             │ 2. Parse (Nemotron)                │ 2. Generate Query
             │ 3. Chunk (Semantic)                │    Embedding
             │ 4. Embed (BGE-M3)                  ▼
             │ 5. Index (ChromaDB)        ┌─────────────────────┐
             ▼                            │   CHROMADB          │
     ┌───────────────────┐                │   (Vector Search)   │
     │   DATABASE        │◄───────────────┤   30 vectors        │
     │   - Documents: 1  │                └─────────┬───────────┘
     │   - Chunks: 11    │                          │
     │   - Pages: 10     │                          │ 3. Retrieve
     │   - Images: 6     │                          │    Top Chunks
     │   - Tables: 3     │                          ▼
     └───────────────────┘                ┌─────────────────────┐
                                          │  RESPONSE GEN       │
                                          │  (Currently: Simple)│
                                          │  (Future: LLM)      │
                                          └─────────┬───────────┘
                                                    │
                                                    │ 4. Return Answer
                                                    │    + Sources
                                                    ▼
                                          ┌─────────────────────┐
                                          │  DATABASE           │
                                          │  - Conversation     │
                                          │  - Messages         │
                                          └─────────────────────┘
```

## 🔄 Pipeline RAG Completo

### En el Admin Panel (Procesamiento)

```
Documento PDF/DOCX
        ↓
[1] PARSING (10-30%)
    - Nemotron Parse v1.1
    - Extrae: texto, imágenes, tablas
    - Genera páginas anotadas (PNG)
        ↓
[2] CHUNKING (30-50%)
    - Semantic chunking
    - Tamaño: 1200 chars
    - Overlap: 150 chars
    - Max: 4800 chars
        ↓
[3] EMBEDDINGS (50-75%)
    - BGE-M3 (1024 dims)
    - GPU acceleration
    - Batch processing
        ↓
[4] INDEXING (75-100%)
    - ChromaDB storage
    - Metadata preservation
        ↓
    ✓ Documento Completado
```

### En el Chatbot (Consulta)

```
Pregunta del Usuario
        ↓
1. Generate Query Embedding
   (BGE-M3, same model as docs)
        ↓
2. Vector Search in ChromaDB
   (Cosine similarity)
   Top 5 chunks retrieved
        ↓
3. [Optional] Reranking
   (BGE-reranker - future)
        ↓
4. Generate Response
   - Current: Show chunks
   - Future: LLM integration
        ↓
5. Return with Sources
   - Document names
   - Chunk previews
        ↓
    Save to Conversation
```

## 📁 Estructura de Datos

### Admin Panel
```
Document
├── Pages (10) → annotated PNG images with bounding boxes
├── Images (6) → extracted figures
├── Tables (3) → extracted tables as images
├── Chunks (11) → text fragments with embeddings
└── Logs → processing history
```

### Chatbot
```
Conversation (session-based)
└── Messages
    ├── User messages
    └── Assistant responses
        └── Retrieved Chunks (ManyToMany)
```

## 🎨 Diseño de Interfaz

### Admin Panel
- **Colores:** Moderno con gradientes índigo-violeta (#6366f1, #8b5cf6)
- **Layout:** Sidebar + Main Content
- **Cards:** Estadísticas con gradientes sutiles
- **Progress Bars:** Animados con gradientes
- **Páginas Anotadas:** Visualización lado a lado (texto + PNG)

### Chatbot
- **Colores:** Mismo esquema de gradientes
- **Layout:** Full-height chat interface
- **Burbujas:** Usuario (gradiente) vs Asistente (blanco)
- **Iconos:** Usuario (person) vs IA (robot)
- **Animaciones:** Slide-in para mensajes
- **Fuentes:** Cards con borde izquierdo coloreado

## 🚀 Estado Actual

### ✅ Completado
- [x] Panel de administración funcional
- [x] Pipeline RAG completo (4 etapas)
- [x] Procesamiento en background con Celery
- [x] Visualización de páginas anotadas
- [x] Dashboard con estadísticas
- [x] Chatbot con interfaz moderna
- [x] Búsqueda vectorial funcional
- [x] Sistema de conversaciones
- [x] Muestra de fuentes
- [x] GPU acceleration (RTX 5080)

### 🔄 En Progreso
- [ ] Integración con LLM (OpenAI/Llama/Mistral)
- [ ] Reranking con BGE-reranker
- [ ] Memoria de conversación (contexto)

### 📝 Próximos Pasos

#### 1. Integración con LLM (Prioridad Alta)
```python
# Ejemplo con OpenAI
def generate_response_with_llm(query, chunks):
    context = "\n\n".join([chunk.text for chunk in chunks])
    
    prompt = f"""Basándote en el siguiente contexto, responde la pregunta.
    
    Contexto:
    {context}
    
    Pregunta: {query}
    
    Respuesta:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente experto."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content
```

#### 2. Reranking (Prioridad Media)
```python
from reranker import Reranker

def rerank_results(query, chunks):
    reranker = Reranker()
    scores = reranker.rank(query, [c.text for c in chunks])
    return [chunks[i] for i in scores.argsort()[::-1]]
```

#### 3. Filtros Avanzados (Prioridad Baja)
- Filtrar por documento específico
- Filtrar por rango de fechas
- Filtrar por tipo de contenido

## 📊 Métricas del Sistema

### Base de Datos
- **Documentos procesados:** 1
- **Páginas extraídas:** 10
- **Imágenes extraídas:** 6
- **Tablas extraídas:** 3
- **Chunks generados:** 11
- **Vectores en ChromaDB:** 30

### Configuración
- **Chunk size:** 1200 caracteres
- **Overlap:** 150 caracteres
- **Max chunk size:** 4800 caracteres
- **Embedding dimension:** 1024 (BGE-M3)
- **Vector search:** Top 5 results
- **GPU:** NVIDIA GeForce RTX 5080

## 🎯 Casos de Uso

### Caso 1: Consulta Simple
**Usuario:** "¿Cuáles son las propiedades de la turbina?"  
**Sistema:** 
1. Genera embedding de la pregunta
2. Busca en ChromaDB (cosine similarity)
3. Recupera 5 chunks relevantes
4. Muestra respuesta con fuentes

### Caso 2: Consulta Específica
**Usuario:** "¿Qué dimensiones tiene el rotor?"  
**Sistema:**
1. Mismo proceso
2. Chunks más específicos (mayor similitud)
3. Referencias exactas al documento

### Caso 3: Conversación con Contexto (Futuro)
**Usuario:** "¿Cuáles son las propiedades?"  
**Usuario:** "¿Y las dimensiones de eso?"  
**Sistema:** Mantiene contexto de "propiedades" mencionadas antes

## 🔧 Tecnologías Utilizadas

### Backend
- **Django 5.0** - Framework web
- **Celery 5.3.6** - Procesamiento asíncrono
- **Redis** - Message broker
- **SQLite** - Base de datos

### ML/AI
- **NVIDIA Nemotron Parse v1.1** - Document parsing
- **BGE-M3** - Embeddings (1024 dims)
- **ChromaDB 0.5.23** - Vector database
- **PyTorch** - Deep learning framework

### Frontend
- **Bootstrap 5.3.0** - UI framework
- **Bootstrap Icons** - Iconografía
- **Custom CSS** - Diseño moderno con gradientes
- **Vanilla JavaScript** - Interactividad

### Hardware
- **GPU:** NVIDIA GeForce RTX 5080
- **CUDA:** Para aceleración de embeddings

## 📚 Documentación Adicional

- **Admin Panel:** `WebApp/admin_panel/README.md`
- **Chatbot:** `WebApp/chatbot/README.md`
- **General:** `WebApp/README.md`
- **Módulos RAG:** `README_*.md` en raíz del proyecto

## 🎉 Conclusión

El sistema está **completamente funcional** y listo para consultas. Los usuarios pueden:

1. **Administradores:** Subir y procesar documentos
2. **Público:** Hacer preguntas y recibir respuestas con fuentes

**Próximo gran paso:** Integración con LLM para generar respuestas más naturales y conversacionales.
