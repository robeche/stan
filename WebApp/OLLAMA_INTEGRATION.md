# 🤖 Integración Ollama Completada

## ✅ Qué se hizo

### 1. Configuración
- Detectado modelo: **gpt-oss:20b** (12.85 GB)
- URL: **http://localhost:11434**
- Configurado en `settings.py`

### 2. Código del Chatbot Actualizado
El archivo [chatbot/views.py](chatbot/views.py) ahora:

1. **Genera embedding** de la pregunta con BGE-M3
2. **Busca en ChromaDB** los 5 chunks más relevantes
3. **Construye un prompt** con el contexto de los chunks
4. **Llama a Ollama** para generar una respuesta natural
5. **Devuelve la respuesta** con las fuentes

### 3. Prompt Optimizado para RAG
```python
prompt = f"""Eres un asistente experto que responde preguntas basándose en documentos técnicos.

Contexto de los documentos:
{context}

Pregunta del usuario: {query}

Instrucciones:
- Responde de manera clara y concisa
- Usa solo la información del contexto proporcionado
- Si la información no está en el contexto, indícalo claramente
- Cita el documento cuando sea relevante

Respuesta:"""
```

## 🚀 Cómo Usar

### 1. Verificar Ollama
```bash
python test_ollama.py
```

### 2. Probar el Chatbot
```bash
python test_chatbot.py
```

### 3. Usar la Interfaz Web
```
http://localhost:8000/
```

## ⚙️ Configuración (settings.py)

```python
OLLAMA_CONFIG = {
    'URL': 'http://localhost:11434',
    'MODEL': 'gpt-oss:20b',  # Tu modelo actual
    'TEMPERATURE': 0.7,      # Creatividad (0.0-1.0)
    'TOP_P': 0.9,            # Nucleus sampling
    'TIMEOUT': 60,           # Timeout en segundos
}
```

### Cambiar Modelo

Si descargas otro modelo en Ollama:
```bash
# En tu contenedor Docker
docker exec -it <container> ollama pull llama3.2
docker exec -it <container> ollama pull mistral
```

Luego actualiza `settings.py`:
```python
'MODEL': 'llama3.2',  # o 'mistral', etc.
```

## 📊 Flujo Completo

```
Usuario: "¿Cuáles son las propiedades de la turbina?"
    ↓
1. BGE-M3 genera embedding de la pregunta
    ↓
2. ChromaDB busca chunks similares (cosine similarity)
    ↓
3. Se recuperan los top 3 chunks más relevantes
    ↓
4. Se construye prompt con contexto:
   """
   Contexto:
   - Chunk 1: "The NREL offshore 5-MW baseline wind turbine..."
   - Chunk 2: "Gross Properties: Hub Height 90m, Rotor Diameter 126m..."
   - Chunk 3: "Three-bladed upwind variable-speed..."
   
   Pregunta: ¿Cuáles son las propiedades de la turbina?
   """
    ↓
5. Ollama (gpt-oss:20b) genera respuesta natural
    ↓
6. Usuario recibe:
   - Respuesta en lenguaje natural
   - Referencias a los documentos
   - Chunks usados como fuentes
```

## 🎯 Ventajas de Esta Implementación

### ✅ Local y Privado
- No envía datos a servicios externos
- Gratis (sin API keys)
- Control total sobre el modelo

### ✅ RAG Completo
- Búsqueda vectorial con embeddings
- Contexto relevante para el LLM
- Fuentes verificables

### ✅ Respuestas Naturales
- Ya no muestra chunks crudos
- Lenguaje conversacional
- Cita documentos cuando es relevante

## 📝 Ejemplo de Interacción

**Antes (sin LLM):**
```
Encontré 5 fragmentos relevantes en la base de datos.

Resumen de los documentos relacionados:

1. De 'NREL5MW_Reduced':
The NREL offshore 5-MW baseline wind turbine is a conventional
three-bladed upwind variable-speed variable blade-pitch-to-feather-
controlled turbine...
```

**Ahora (con Ollama):**
```
La turbina NREL 5MW es una turbina eólica offshore convencional con las 
siguientes características principales:

- Configuración: Tres palas, orientación contra el viento (upwind)
- Control: Velocidad variable con pitch-to-feather
- Altura del buje: 90 metros
- Diámetro del rotor: 126 metros
- Potencia nominal: 5 MW

Según el documento NREL5MW_Reduced, esta turbina está diseñada como 
línea base para estudios de turbinas offshore de gran escala.

[Fuentes consultadas en los cards debajo]
```

## ⚡ Rendimiento

- **Primera consulta:** ~10-30 segundos (carga del modelo)
- **Consultas siguientes:** ~5-15 segundos (modelo en memoria)
- **Depende de:**
  - Tamaño del modelo (gpt-oss:20b = 12.85 GB)
  - Hardware del servidor Docker
  - Longitud de la respuesta

## 🔧 Troubleshooting

### Error: "No se pudo conectar al servidor Ollama"
```bash
# Verifica que el contenedor esté corriendo
docker ps | grep ollama

# Si no está corriendo, inícialo
docker start <container_name>
```

### Error: "Model not found"
```bash
# Lista modelos disponibles
docker exec <container> ollama list

# Descarga el modelo si no existe
docker exec <container> ollama pull gpt-oss:20b
```

### Respuestas muy lentas
- Considera usar un modelo más pequeño (llama3.2:3b en lugar de 20b)
- Verifica recursos del contenedor Docker
- Ajusta `TIMEOUT` en settings.py si es necesario

## 🎉 Resultado

¡Tu chatbot RAG ahora tiene IA conversacional integrada!

- ✅ Busca información en tus documentos
- ✅ Genera respuestas naturales y contextualizadas
- ✅ Cita fuentes
- ✅ 100% local y privado
