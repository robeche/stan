f# Guión para Video - Sistema RAG de Documentación Técnica

## 1. INTRODUCCIÓN PERSONAL (30-45 segundos)

**[Apertura con tu presentación en cámara]**

Hola, mi nombre es Roberto Echeverría Delgado, soy Ingeniero de control de aerogeneradores. Toda mi carrera ha estado relacionada con el desarrollo de nuevas tecnologías relacionadas con los molinos... más grandes, más potentes, más ligeros, pero siempre desde la perspectiva del control.

A lo largo de mi carrera, he tenido la oportunidad de aplicar desde la ingeniería de control básica hasta técnicas más punteras que integran "Machine learning" dentro de los lazos de control, como pueden ser MPC (Model Predictive Control), el uso de sistemas LIDAR para control FeedForward, control multivariable en aerogeneradores flotantes. 

Además de la ingeniería de aerogeneradores, tengo otra gran inquietud o hobbie, como es el desarrollo de videojuegos, siempre relacionados con la simulación y el aprendizaje.

Sin embargo, para este proyecto he preferido dejar a un lado mi experiencia y mi hobby para adentrarme en un área totalmente diferente como es el tratamiento de documentación. A lo largo de mi experiencia laboral me he dado cuenta de que en este área, los ingenieros nos encontramos con una barrera importante a la hora de gestionar la enorme cantidad de documentos, informes, correos electrónicos, etc... Ante esto, los LLMs han supuesto un tremendo alivio pudiendo compartir dicha documentación con los chatbots y haciéndoles preguntas... sin embargo, tenemos el problema de la confidencialidad y los copyrights... en muchas ocasiones no vamos a poder, o bueno... no deberíamos compartir cierta documentación con los chatbots, incluso si nuestra empresa integra sistemas de pago... ya que los clientes quizás lo impiden expresamente...

Por esta razón, he creado un sistema propio, que me va a permitir mantener mi propio chatbot local y alimentarlo con la documentación confidencial necesaria. Para ello he construido una aplicación basada en un **Sistema de RAGS** diseñado específicamente para documentación técnica. 

Esta aplicación permite:
- **Procesar automáticamente** documentos PDF técnicos complejos
- **Extraer** tablas, figuras y ecuaciones
- **Realizar consultas en lenguaje natural** sobre la documentación
- **Obtener respuestas precisas** con referencias a las fuentes originales

El sistema está diseñado para resolver uno de los mayores desafíos en ingeniería: encontrar información específica en extensos manuales técnicos y documentación de proyectos, así como buscar referencias en normas cuyo copyright impide el uso fuera de los sistemas de la empresa.

Además, busco con el sistema devuelva una respuesta multimodal, para ello he conseguido separar los contenidos como tablas, figuras y ecuaciones y guardarlos con las referencias adecuadas, de manera que si el chatbot requiere mostrarlas estas se devolverán en el formato original indicando la referencia adecuada.

---

## 3. ARQUITECTURA DEL SISTEMA (2-3 minutos)

**[Mostrar el diagrama de arquitectura]**

A continuación explicaré la arquitectura de la aplicación.

He estructurado el sistema en tres capas principales y dos sistemas de almacenamiento.

En primer lugar, la  **CAPA DE PRESENTACIÓN**

La aplicación cuenta con dos interfaces principales:

1. **un Panel de administración**: Donde los usuarios pueden:
   - Subir documentos PDF
   - Monitorear el proceso de análisis
   - Visualizar páginas, tablas e imágenes extraídas
   - Gestionar el contenido indexado

2. y una ventana con el **Chatbot**: Una interfaz conversacional donde:
   - Los usuarios realizan preguntas en lenguaje natural
   - El sistema responde con información precisa
   - Se muestran las fuentes y referencias visuales

En segundo lugar tenemos la **CAPA DE COORDINACIÓN**

Esta capa es el cerebro operativo del sistema:

- **Django**: Framework web que gestiona las peticiones, la autenticación y la lógica de negocio. Básicamente, he elegido Django porque tengo experiencia con él y... más vale malo conocido que bueno por conocer. 
- **Celery**: Sistema de tareas asíncronas que procesa documentos en segundo plano, permitiendo manejar archivos grandes sin bloquear la aplicación... El objetivo es que en una futura mejora, pueda generar una base de datos de todos los documentos y normativas de la empresa y dejar el sistema procesando en segundo plano.
- **Redis**: Almacén en memoria que actúa como broker de mensajes para Celery, gestionando la cola de tareas...

### **CAPA DE PROCESAMIENTO**

Aquí es donde ocurre la magia del análisis de documentos:

1. **Parsing con Nemotron**: 
   - Utilizamos el modelo NVIDIA Nemotron-Parse-v1.1
   - Este modelo de IA extrae texto, detecta tablas, figuras y fórmulas
   - Preserva la estructura visual del documento
   - Genera markdown con anotaciones de coordenadas
   Empecé utilizando otros modelos, de los explicados en el curso, pero en una de las charlas de Patxi mencionó que NVIDIA habia liberado este modelo e inmediatamente me fui a probarlo... Con los otros modelos había tenido muchos problemas para parsear las tablas y las ecuaciones. Me costó un poco echarlo a andar, más que nada porque las imágenes de docker que hay disponibles no soportan bien las RTX5000... (no entiendo por qué...) y yo tengo una RTX5080... La cuestión es que estaba a punto de tirar la toalla hasta que ví que era posible hacerlo funcionar directamente con el modelo de HuggingFace, así que me puse con ello y funcionó... a pesar de que también me costó un poco hacer funcionar pytorch con la RTX5080... pero al final resultó que tenía que usar python 3.10 en lugar de 3.13... pero eso ya es otro tema...

2. **Chunking**:
   - El documento se divide en fragmentos manejables... aquí todavía me queda enredar un poco el tamaño óptimo de los chunks para que el llm local no se sature con el tamaño del contexto.
   - La idea es que cada chunk mantiene contexto coherente
   - También se preservan metadatos de ubicación para la utilización posterior de las tablas y figuras 

3. **Embeddings**:
   - Generamos vectores semánticos con BGE-M3, de la misma manera que lo hacíamos en los ejemplos del curso.
   - Cada fragmento se convierte en una representación numérica permitiendo búsquedas por similitud semántica

4. **LLM (Large Language Model)**:
   - Utilizamos Ollama para generar respuestas y el modelo gpt:oss con 20billones de parámetros.
   - El modelo recibe el contexto relevante recuperado
   - Genera respuestas naturales y precisas... 

### **ALMACENAMIENTO**

El sistema utiliza dos bases de datos complementarias:

1. **SQLite (Metadatos)**:
   - Almacena información estructurada: documentos, páginas, tablas, imágenes
   - Gestiona usuarios y conversaciones
   - Mantiene el registro de procesamiento

2. **ChromaDB (Vectores)**:
   - Base de datos vectorial especializada
   - Almacena los embeddings de cada fragmento
   - Permite búsquedas rápidas por similitud semántica

---

## 4. FLUJO DE PROCESAMIENTO (1-2 minutos)

**[Demostración práctica subiendo un documento]**

Veamos cómo funciona el proceso completo:

**Paso 1: Carga del Documento**
- El usuario sube un PDF desde el Admin Panel
- Django registra el documento en SQLite
- Celery recibe la tarea de procesamiento

**Paso 2: Parsing**
- Nemotron analiza cada página del PDF
- Extrae texto, detecta elementos visuales
- Genera archivos markdown con anotaciones
- Guarda imágenes de tablas y figuras

**Paso 3: Fragmentación**
- El documento procesado se divide en chunks
- Cada fragmento mantiene coherencia semántica
- Se preservan metadatos de origen

**Paso 4: Vectorización**
- BGE-M3 genera embeddings para cada chunk
- Los vectores se almacenan en ChromaDB
- Se indexan para búsquedas rápidas

**Paso 5: Consulta**
- El usuario hace una pregunta en el Chatbot
- Se genera un embedding de la consulta
- ChromaDB busca los fragmentos más similares
- El LLM genera una respuesta con ese contexto
- Se muestran las fuentes y elementos visuales relevantes

---

## 5. DEMOSTRACIÓN PRÁCTICA (1-2 minutos)

**[Mostrar el Chatbot en acción]**

Por ejemplo, si pregunto: "¿Cuál es el diámetro del rotor de la turbina NREL 5MW?"

El sistema:
1. Busca en ChromaDB los fragmentos relevantes
2. Encuentra referencias a la Tabla 1-1
3. El LLM genera una respuesta precisa
4. Automáticamente muestra la tabla referenciada
5. Proporciona la respuesta: 126 metros

Todo esto en pocos segundos, con referencias verificables.

---

## 6. VENTAJAS Y CASOS DE USO (30-45 segundos)

**Ventajas clave:**
- ✅ Procesamiento automático de documentos técnicos complejos
- ✅ Búsquedas en lenguaje natural, sin necesidad de conocer la estructura del documento
- ✅ Respuestas con referencias verificables
- ✅ Extracción y visualización automática de tablas y figuras
- ✅ Procesamiento en segundo plano para documentos grandes

**Casos de uso:**
- Consulta de manuales técnicos de ingeniería
- Análisis de especificaciones de proyectos
- Búsqueda en documentación de turbinas eólicas
- Acceso rápido a normativas y estándares técnicos

---

## 7. CIERRE (45-60 segundos)

**[Vuelta a cámara]**

Y bueno... ¿qué os puedo decir? Este ha sido un viaje bastante intenso, la verdad. Al principio pensaba que iba a ser más sencillo, pero cuando te metes de lleno en el mundo de los RAGs te das cuenta de que hay muchísimas piezas que tienen que encajar perfectamente.

Lo que más me ha gustado de este proyecto es que realmente resuelve un problema del día a día. Todos los que trabajamos con documentación técnica sabemos lo frustrante que es perder horas buscando ese dato específico entre cientos de páginas... Ahora simplemente preguntas y ya está. Es como tener un compañero que se ha leído toda la documentación y tiene memoria fotográfica.

Obviamente, todavía hay margen de mejora. Me gustaría optimizar mejor el tamaño de los chunks, añadir más formatos de entrada, quizás integrar OCR para documentos escaneados... pero oye, para ser un primer acercamiento serio al procesamiento de documentación, estoy bastante contento con el resultado.

Y lo mejor de todo es que todo funciona en local. Nada de mandar información confidencial a servicios externos, nada de preocuparte por los copyrights o por lo que diga el cliente. Todo se queda en tu máquina.

Así que nada, si tenéis alguna pregunta, sugerencia o simplemente queréis charlar sobre el proyecto, estaré encantado de hablar. Y si os animáis a montar algo parecido, adelante, que al final se aprende muchísimo peleándose con estos sistemas.

¡Muchas gracias por llegar hasta aquí y nos vemos!

---

## NOTAS DE PRODUCCIÓN

### Duración Total Estimada: 6-8 minutos

### Recursos Visuales Necesarios:
- Tu presentación en cámara (introducción y cierre)
- Captura del diagrama de arquitectura (animado con zoom a cada sección)
- Grabación de pantalla del Admin Panel
- Grabación de pantalla del Chatbot
- Demo de carga y procesamiento de documento
- Demo de consulta con respuesta y visualización de tabla

### Ritmo Sugerido:
- Introducción: Tono profesional y cercano
- Arquitectura: Pausado y didáctico, destacar cada componente
- Demostración: Dinámico, mostrar el sistema en acción
- Cierre: Confiado y profesional

### Tips de Presentación:
- Usa puntero o resaltado en el diagrama para guiar la atención
- Muestra ejemplos reales de consultas
- Destaca los tiempos de respuesta
- Si es posible, muestra el procesamiento en tiempo real (acelerado)
