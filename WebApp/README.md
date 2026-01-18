# Django RAG Document Processing Application

Aplicación web Django con dos módulos principales:
1. **Panel de Administración**: Para subir y procesar documentos
2. **Chatbot Público**: Para consultar los documentos mediante IA conversacional

Pipeline RAG completo:
- **Parsing** con Nemotron Parse v1.1
- **Chunking** inteligente semántico
- **Embeddings** con BGE-M3 (1024 dimensiones)
- **Vector Storage** con ChromaDB
- **Procesamiento en background** con Celery + Redis

## 📋 Características

### 🔐 Panel de Administración (Admin Panel)
Solo accesible para usuarios administradores:
- ✅ Subida de documentos (PDF, DOCX, DOC, TXT, MD)
- ✅ Procesamiento automático en background
- ✅ Indicador de progreso en tiempo real (4 etapas)
- ✅ Visualización de páginas anotadas con bounding boxes
- ✅ Exploración de imágenes, tablas y fragmentos extraídos
- ✅ Logs detallados del procesamiento
- ✅ Dashboard con estadísticas
- ✅ Modelo de datos interactivo
- ✅ Función de reprocesamiento

### 💬 Chatbot Público (Chat Interface)
Interfaz de consulta abierta para todos los usuarios:
- ✅ Interfaz de chat moderna y responsive
- ✅ Sistema de conversaciones con historial
- ✅ Búsqueda vectorial en ChromaDB
- ✅ Respuestas basadas en documentos indexados
- ✅ Muestra fuentes y referencias
- ✅ Sesiones independientes por usuario
- 🔄 Reranking (próximamente)
- 🔄 Integración con LLM (próximamente)

### Pipeline de Procesamiento
1. **Parsing con Nemotron**: Extrae texto, imágenes y tablas del documento
2. **Chunking**: Divide el documento en fragmentos semánticos (1200 chars, overlap 150)
3. **Embeddings**: Genera vectores con BGE-M3 usando GPU
4. **Indexación**: Almacena en ChromaDB para búsqueda vectorial

## 🚀 Instalación y Configuración

### 1. Prerrequisitos

- Python 3.12+ (usar el entorno virtual existente en `../venv`)
- Redis Server (para Celery)
- CUDA GPU (recomendado para embeddings)
- NGC API Key (para Nemotron parsing)

### 2. Instalar Redis

**Windows:**
```powershell
# Opción 1: Chocolatey
choco install redis-64

# Opción 2: Descargar desde https://github.com/microsoftarchive/redis/releases
# Ejecutar redis-server.exe
```

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# macOS
brew install redis
brew services start redis
```

### 3. Instalar dependencias

```powershell
# Activar entorno virtual
..\venv\Scripts\Activate.ps1

# Instalar dependencias Django
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crear archivo `.env` en la carpeta `WebApp`:

```env
NGC_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxx
SECRET_KEY=django-secret-key-change-this-in-production
DEBUG=True
```

### 5. Inicializar base de datos

```powershell
# Crear migraciones
python manage.py makemigrations

# Aplicar migraciones
python manage.py migrate

# Crear superusuario
python manage.py createsuperuser
```

### 6. Recopilar archivos estáticos

```powershell
python manage.py collectstatic --noinput
```

## 🎯 Uso

### Iniciar la aplicación

**Terminal 1 - Django Development Server:**
```powershell
..\venv\Scripts\Activate.ps1
python manage.py runserver
```

**Terminal 2 - Celery Worker:**
```powershell
..\venv\Scripts\Activate.ps1
celery -A rag_project worker -l info --pool=solo
```

**Terminal 3 - Redis Server** (si no está ejecutándose como servicio):
```powershell
redis-server
```

### Acceder a la aplicación

1. **Panel de Administración**: http://localhost:8000/admin-panel/
   - Requiere autenticación como staff/superuser
   - Dashboard con estadísticas
   - Subir documentos
   - Ver procesamiento en tiempo real

2. **Django Admin**: http://localhost:8000/admin/
   - Gestión avanzada de modelos
   - Configuración de usuarios

## 📊 Estructura del Proyecto

```
WebApp/
├── manage.py                      # Django management script
├── requirements.txt               # Dependencias Python
├── README.md                      # Esta documentación
├── test_chatbot.py                # Script de prueba del chatbot
│
├── rag_project/                   # Proyecto Django principal
│   ├── __init__.py               # Configuración Celery
│   ├── settings.py               # Configuración Django
│   ├── urls.py                   # URLs principales
│   ├── celery.py                 # Configuración Celery
│   ├── wsgi.py                   # WSGI entry point
│   └── asgi.py                   # ASGI entry point
│
├── admin_panel/                   # 🔐 Módulo 1: Panel de Administración
│   ├── models.py                 # Modelos: Document, Page, Image, Table, Chunk, ProcessingLog
│   ├── views.py                  # Vistas del panel admin
│   ├── urls.py                   # URLs: /admin-panel/*
│   ├── forms.py                  # Formularios
│   ├── tasks.py                  # Tareas Celery (pipeline RAG)
│   ├── admin.py                  # Configuración Django Admin
│   └── apps.py                   # Configuración app
│
├── chatbot/                       # 💬 Módulo 2: Interfaz de Chat Público
│   ├── models.py                 # Modelos: Conversation, Message
│   ├── views.py                  # Vistas del chatbot
│   ├── urls.py                   # URLs: /* (raíz)
│   ├── admin.py                  # Configuración Django Admin
│   ├── apps.py                   # Configuración app
│   └── README.md                 # Documentación del chatbot
│
├── templates/                     # Templates HTML
│   ├── admin_panel/
│   │   ├── base.html             # Template base admin
│   │   ├── dashboard.html        # Dashboard con estadísticas
│   │   ├── document_list.html    # Lista de documentos
│   │   ├── document_upload.html  # Subir documento
│   │   └── document_detail.html  # Detalle, progreso, páginas anotadas
│   └── chatbot/
│       └── chat.html             # Interfaz de chat moderna
│
├── tools/                         # Módulos RAG compartidos
│   ├── parser.py                 # Nemotron parsing
│   ├── chunker.py                # Semantic chunking
│   ├── embeddings.py             # BGE-M3 embeddings
│   └── vector_store.py           # ChromaDB interface
│
├── media/                         # Archivos subidos y generados
│   ├── documents/                # Documentos originales
│   ├── annotated_pages/          # Páginas con bounding boxes (PNG)
│   ├── extracted_images/         # Imágenes extraídas
│   └── extracted_tables/         # Tablas extraídas
│
├── processing_output/             # Output temporal del procesamiento
│   └── {document_name}/
│       ├── raw_output/           # Texto raw de cada página
│       └── annotated_pages/      # PNGs anotados originales
│
├── chroma_db/                     # Base de datos vectorial ChromaDB
└── staticfiles/                   # Archivos estáticos
```

## 🗄️ Modelos de Base de Datos

### 🔐 Admin Panel Models

#### Document
Documento principal con estado de procesamiento, progreso y estadísticas.

**Campos principales:**
- `title`, `original_filename`, `file`
- `status`: uploaded → parsing → chunking → embedding → indexing → completed
- `progress_percentage`: 0-100%
- `celery_task_id`: ID de la tarea Celery
- Flags: `parsing_completed`, `chunking_completed`, `embedding_completed`, `indexing_completed`
- Estadísticas: `total_pages`, `total_chunks`, `total_images`, `total_tables`

### Page
Páginas extraídas del documento con contenido markdown y páginas anotadas (PNG con bounding boxes).

### Image
Imágenes extraídas con metadatos (caption, dimensiones, página).

### Table
Tablas extraídas como imágenes con metadatos (caption, página).

### Chunk
Fragmentos del documento con:
- Contenido textual
- Embedding vector (lista JSON, 1024 dimensiones)
- Metadatos del chunking
- ID de ChromaDB
- Relación con documento

### ProcessingLog
Logs detallados del procesamiento (info, warning, error, success).

### 💬 Chatbot Models

#### Conversation
Sesión de chat con identificador único:
- `session_id`: UUID único por sesión
- `created_at`, `updated_at`: timestamps
- Relación con Messages

#### Message
Mensaje individual en la conversación:
- `message_type`: 'user' o 'assistant'
- `content`: contenido del mensaje
- `retrieved_chunks`: ManyToMany con Chunk (fuentes usadas)
- `created_at`: timestamp

## 🎯 URLs y Rutas

### Panel de Administración (requiere login)
- `/admin-panel/` - Dashboard con estadísticas
- `/admin-panel/documents/` - Lista de documentos
- `/admin-panel/documents/upload/` - Subir nuevo documento
- `/admin-panel/documents/<id>/` - Detalle del documento
- `/admin-panel/documents/<id>/reprocess/` - Reprocesar documento
- `/admin-panel/model-diagram/` - Diagrama del modelo de datos
- `/admin-panel/api/document-status/<id>/` - API de estado (JSON)

### Chatbot (público)
- `/` - Interfaz de chat principal
- `/send/` - API para enviar mensajes (POST)
- `/new/` - Crear nueva conversación (POST)

### Django Admin
- `/admin/` - Panel de administración de Django

## ⚙️ Configuración

### settings.py - Configuración RAG

```python
RAG_CONFIG = {
    'PARSING': {
        'MODEL_NAME': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
        'API_KEY_ENV': 'NGC_API_KEY',
        'OUTPUT_DIR': RAG_MODULES_PATH / 'output_simple',
    },
    'CHUNKING': {
        'STRATEGY': 'hybrid_semantic',
        'CHUNK_SIZE': 512,
        'OVERLAP': 50,
    },
    'EMBEDDINGS': {
        'MODEL_NAME': 'BAAI/bge-m3',
        'DEVICE': 'cuda',  # o 'cpu'
        'BATCH_SIZE': 8,
    },
    'VECTOR_STORE': {
        'COLLECTION_NAME': 'rag_documents',
        'PERSIST_DIRECTORY': str(RAG_MODULES_PATH / 'output_rag' / 'chroma_db'),
    },
}
```

## 🔄 Pipeline de Procesamiento

### Tarea Celery: `process_document`

Ubicación: `admin_panel/tasks.py`

**Etapas:**

1. **Parsing (10-30%)**
   - Llama a `parse_local.py`
   - Extrae páginas, imágenes, tablas
   - Almacena en base de datos

2. **Chunking (30-50%)**
   - Usa `document_chunker.py`
   - Divide en fragmentos semánticos
   - Guarda chunks en BD

3. **Embeddings (50-75%)**
   - Usa `embedding_generator.py` (BGE-M3)
   - Genera vectores de 1024 dimensiones
   - Almacena en chunks

4. **Indexación (75-100%)**
   - Usa `vector_store.py`
   - Indexa en ChromaDB
   - Marca chunks como indexados

**Logs en tiempo real:** Cada etapa genera logs visibles en la interfaz.

## 🔍 API Endpoints

### `/admin-panel/api/task-status/<task_id>/`
Consultar estado de tarea Celery.

**Respuesta:**
```json
{
  "task_id": "abc123",
  "status": "PROGRESS",
  "ready": false,
  "successful": null
}
```

### `/admin-panel/api/document-status/<document_id>/`
Consultar estado de procesamiento de documento.

**Respuesta:**
```json
{
  "document_id": 1,
  "status": "embedding",
  "progress_percentage": 65,
  "parsing_completed": true,
  "chunking_completed": true,
  "embedding_completed": false,
  "indexing_completed": false,
  "total_pages": 24,
  "total_chunks": 24,
  "recent_logs": [...]
}
```

## 🎨 Interfaz de Usuario

### Dashboard
- Estadísticas generales
- Documentos recientes
- Logs recientes
- Auto-refresh si hay documentos procesándose

### Lista de Documentos
- Tabla con todos los documentos
- Filtros por estado
- Progreso visual
- Botones de acción (ver, eliminar)

### Detalle de Documento
- Información completa
- Indicadores de progreso por etapa
- Tabs: Logs, Páginas, Fragmentos, Imágenes, Tablas
- Auto-actualización cada 3 segundos

### Subir Documento
- Formulario simple
- Información del pipeline
- Inicio automático de procesamiento

## 🐛 Troubleshooting

### Redis no se conecta
```bash
# Verificar que Redis está corriendo
redis-cli ping
# Debe responder: PONG

# Windows: ejecutar redis-server.exe
redis-server
```

### Celery no procesa tareas
```bash
# Verificar que el worker está activo
celery -A rag_project inspect active

# En Windows, usar pool solo:
celery -A rag_project worker -l info --pool=solo
```

### Error en parsing (Nemotron)
```bash
# Verificar NGC_API_KEY
echo $env:NGC_API_KEY

# Configurar si no existe
$env:NGC_API_KEY = "nvapi-xxxxx"
```

### Error CUDA out of memory
Cambiar en `settings.py`:
```python
'DEVICE': 'cpu',  # en lugar de 'cuda'
```

### Migraciones no aplicadas
```bash
python manage.py makemigrations admin_panel
python manage.py migrate
```

## 📝 Próximos Pasos

### Aplicación de Consulta (Módulo 2)
- Interface pública para hacer consultas
- Búsqueda vectorial en ChromaDB
- Reranking con BGE-reranker
- Generación de respuestas con LLM
- Historial de consultas

### Mejoras Potenciales
- [ ] Soporte para PostgreSQL
- [ ] WebSockets para updates en tiempo real
- [ ] Gestión de múltiples colecciones ChromaDB
- [ ] Preview de imágenes y tablas en UI
- [ ] Exportar chunks y embeddings
- [ ] Búsqueda de documentos
- [ ] Gestión de permisos por usuario
- [ ] API REST completa

## 📚 Referencias

- **Django**: https://docs.djangoproject.com/
- **Celery**: https://docs.celeryq.dev/
- **ChromaDB**: https://docs.trychroma.com/
- **BGE-M3**: https://huggingface.co/BAAI/bge-m3
- **Módulos RAG**: Ver READMEs en directorio raíz

## 🤝 Soporte

Para problemas relacionados con:
- **Django/Celery**: Revisar logs en consola
- **Pipeline RAG**: Ver logs de procesamiento en interfaz
- **Modelos**: Consultar READMEs de módulos individuales

---

**Autor:** Sistema RAG Document Processing  
**Versión:** 1.0.0  
**Fecha:** Enero 2026
