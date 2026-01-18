# Chatbot RAG - Interfaz de Consulta

## Descripción

La aplicación de chatbot proporciona una interfaz pública para que los usuarios hagan preguntas sobre los documentos indexados en el sistema RAG.

## Características

### 1. **Interfaz de Chat Moderna**
- Diseño limpio y profesional con gradientes modernos
- Mensajes diferenciados para usuario y asistente
- Auto-scroll y animaciones suaves
- Responsive y optimizado para diferentes dispositivos

### 2. **Sistema de Conversaciones**
- Cada sesión mantiene su propio historial de conversación
- Posibilidad de iniciar nuevas conversaciones
- Las conversaciones se guardan en la base de datos

### 3. **Pipeline RAG Completo**
El chatbot implementa el flujo completo de RAG:

```
Usuario → Query → Embedding → ChromaDB Search → Reranking → LLM → Respuesta
```

**Paso 1: Generación de Embeddings**
- El mensaje del usuario se convierte en un vector usando BGE-M3
- Mismo modelo usado para generar los embeddings de los documentos

**Paso 2: Búsqueda Vectorial**
- Se consulta ChromaDB con el embedding de la query
- Recupera los top 5 chunks más relevantes

**Paso 3: Reranking** (opcional, puede implementarse)
- Los resultados pueden ser reordenados usando BGE-reranker
- Mejora la precisión de los resultados

**Paso 4: Generación de Respuesta**
- Actualmente: respuesta simple mostrando los chunks recuperados
- Futuro: integración con LLM (OpenAI, Llama, etc.) para generar respuestas naturales

### 4. **Fuentes y Referencias**
- Cada respuesta muestra las fuentes consultadas
- Links a los documentos originales
- Previews de los fragmentos relevantes

## Modelos de Datos

### Conversation
Representa una sesión de chat:
- `session_id`: Identificador único de la sesión
- `created_at`: Fecha de creación
- `updated_at`: Última actualización

### Message
Representa un mensaje individual:
- `conversation`: Referencia a la conversación
- `message_type`: 'user' o 'assistant'
- `content`: Contenido del mensaje
- `retrieved_chunks`: Chunks usados para generar la respuesta (ManyToMany)

## URLs

- `/` - Interfaz principal del chatbot (raíz del sitio)
- `/send/` - API endpoint para enviar mensajes (POST)
- `/new/` - Crear nueva conversación (POST)

## Uso

### Acceso a la Interfaz

1. **Visita la URL raíz**: `http://localhost:8000/`
2. **Escribe tu pregunta** en el campo de texto
3. **Presiona Enter o clic en "Enviar"**
4. **Recibe una respuesta** con las fuentes consultadas

### Ejemplos de Preguntas

```
¿Cuáles son las propiedades estructurales de la turbina NREL5MW?

¿Qué información hay sobre el rotor?

¿Cuáles son las dimensiones de las palas?
```

### API Usage

#### Enviar Mensaje

```javascript
fetch('/send/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
        message: "Tu pregunta aquí" 
    })
})
.then(response => response.json())
.then(data => {
    console.log('Respuesta:', data.assistant_message.content);
    console.log('Fuentes:', data.assistant_message.sources);
});
```

#### Nueva Conversación

```javascript
fetch('/new/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    }
})
.then(() => location.reload());
```

## Configuración

### Dependencias

Las siguientes herramientas del módulo RAG se utilizan:
- `tools.embeddings.EmbeddingGenerator` - Para generar embeddings de las queries
- `tools.vector_store.VectorStore` - Para buscar en ChromaDB
- `admin_panel.models.Chunk` - Para acceder a los chunks de los documentos

### Variables de Entorno

No se requieren variables adicionales. El chatbot usa:
- La misma ChromaDB que el panel de administración
- Los mismos modelos de embeddings (BGE-M3)

## Mejoras Futuras

### 1. **Integración con LLM**
Actualmente, el chatbot solo muestra los chunks recuperados. Se puede integrar con:
- OpenAI GPT-4/3.5
- Llama 2/3
- Mistral
- Claude

### 2. **Reranking**
Implementar el reranking con BGE-reranker para mejorar la calidad de los resultados.

### 3. **Memoria de Conversación**
Mantener el contexto de la conversación para preguntas de seguimiento.

### 4. **Filtros Avanzados**
- Filtrar por documento específico
- Filtrar por rango de fechas
- Filtrar por tipo de contenido (texto, tablas, figuras)

### 5. **Citas Inline**
Agregar citas numeradas inline en la respuesta: `[1]`, `[2]`, etc.

### 6. **Export de Conversaciones**
Permitir exportar el historial de chat a PDF o TXT.

### 7. **Feedback de Usuarios**
Sistema de thumbs up/down para mejorar las respuestas.

## Arquitectura

```
┌──────────────────────────────────────────────────┐
│                  Usuario                         │
└───────────────────┬──────────────────────────────┘
                    │
                    │ Pregunta
                    ▼
┌──────────────────────────────────────────────────┐
│            chatbot/views.py                      │
│  - send_message()                                │
│  - generate_response()                           │
└───────────────────┬──────────────────────────────┘
                    │
                    ├─────► EmbeddingGenerator
                    │       (BGE-M3)
                    │
                    ├─────► VectorStore
                    │       (ChromaDB Query)
                    │
                    ├─────► Chunk.objects
                    │       (Database)
                    │
                    └─────► [Futuro: LLM]
                            (Generación de respuesta)
```

## Troubleshooting

### Error: "No encontré información relevante"
- Verifica que hay documentos procesados en el panel de administración
- Verifica que ChromaDB tiene chunks indexados
- Intenta reformular la pregunta

### Error al enviar mensaje
- Verifica que el servidor Django está corriendo
- Verifica que Redis y Celery están activos (aunque no son necesarios para el chatbot)
- Revisa los logs del servidor

### Respuestas lentas
- La primera query puede ser lenta (carga del modelo BGE-M3)
- Queries subsecuentes deberían ser más rápidas (modelo en memoria)
- Verifica que CUDA está disponible para GPU acceleration

## Desarrollo

### Agregar un nuevo LLM

1. Instalar el SDK del LLM (ej: `openai`)
2. Agregar la configuración en `settings.py`
3. Modificar `generate_response()` en `views.py`:

```python
def generate_response(query):
    # ... código existente para recuperar chunks ...
    
    # Construir el prompt
    context = "\n\n".join([chunk.text for chunk in chunks])
    prompt = f"""Basándote en el siguiente contexto, responde la pregunta.

Contexto:
{context}

Pregunta: {query}

Respuesta:"""
    
    # Llamar al LLM
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en documentos técnicos."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content, chunks
```

## Licencia

Este módulo es parte del proyecto RAG Document Processing.
