# Memoria del Proyecto RAG - Estructura Modular

Este documento LaTeX ha sido modularizado para facilitar su mantenimiento y edición.

## Estructura de Archivos

### Archivo Principal
- **memoria_proyecto_modular.tex**: Archivo principal que contiene el preámbulo, configuración de paquetes y la inclusión de todos los capítulos.

### Capítulos Individuales
Cada capítulo se encuentra en un archivo separado:

1. **capitulo01_introduccion.tex**
   - Resumen ejecutivo
   - Características principales
   - Objetivos del sistema
   - Stack tecnológico

2. **capitulo02_arquitectura.tex**
   - Visión general del sistema
   - Flujo de procesamiento de documentos
   - Flujo de consulta del chatbot
   - Diagramas TikZ

3. **capitulo03_modulos.tex**
   - Admin Panel
   - Chatbot
   - Parsing (Nemotron)
   - Chunking semántico
   - Embeddings (BGE-M3)
   - Vector Store (ChromaDB)
   - LLM (Ollama)

4. **capitulo04_despliegue.tex**
   - Requisitos del sistema
   - Instrucciones de instalación
   - Configuración

5. **capitulo05_uso.tex**
   - Panel de administración
   - Uso del chatbot

6. **capitulo06_diagramas.tex**
   - Diagramas de secuencia
   - Procesamiento de documento
   - Consulta en chatbot

7. **capitulo07_rendimiento.tex**
   - Métricas de rendimiento
   - Optimizaciones implementadas
   - Recomendaciones

8. **capitulo08_resolucion_problemas.tex**
   - Problemas comunes y soluciones

9. **capitulo09_conclusiones.tex**
   - Logros del proyecto
   - Trabajo futuro
   - Reflexión final

10. **capitulo10_apendices.tex**
    - Comandos útiles
    - Estructura de directorios
    - Referencias

## Compilación

### Opción 1: Usar MiKTeX (Windows)
```bash
pdflatex memoria_proyecto_modular.tex
pdflatex memoria_proyecto_modular.tex  # Segunda pasada para referencias
```

### Opción 2: Con ruta completa de MiKTeX
```powershell
& "C:\Users\<TU_USUARIO>\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" memoria_proyecto_modular.tex
```

### Nota Importante
Es necesario ejecutar `pdflatex` **dos veces** para que:
- La tabla de contenidos se genere correctamente
- Las referencias internas funcionen
- Los números de página se actualicen

## Ventajas de la Estructura Modular

1. **Facilidad de edición**: Cada capítulo puede editarse independientemente sin afectar a los demás
2. **Control de versiones**: Git muestra cambios más claros en archivos pequeños
3. **Colaboración**: Múltiples personas pueden trabajar en diferentes capítulos simultáneamente
4. **Organización**: Estructura clara y fácil de navegar
5. **Compilación selectiva**: Se puede comentar capítulos específicos si es necesario

## Modificar la Estructura

### Agregar un nuevo capítulo
1. Crear un nuevo archivo `capituloXX_nombre.tex`
2. Escribir el contenido comenzando con `\chapter{Título}`
3. Agregar `\include{capituloXX_nombre}` en `memoria_proyecto_modular.tex`

### Eliminar un capítulo
1. Comentar o eliminar la línea `\include{capituloXX_nombre}` en el archivo principal

### Reordenar capítulos
Cambiar el orden de las líneas `\include{}` en el archivo principal

## Archivo Original

El archivo original monolítico `memoria_proyecto.tex` se mantiene como backup pero se recomienda usar la versión modular para futuras ediciones.

## Paquetes LaTeX Utilizados

- **report**: Clase de documento
- **inputenc**: Codificación UTF-8
- **babel**: Idioma español
- **tikz**: Diagramas y gráficos
- **listingsutf8**: Bloques de código con UTF-8
- **hyperref**: Enlaces e hipervínculos
- **geometry**: Márgenes de página
- **fancyhdr**: Encabezados y pies de página
- **tcolorbox**: Cajas de color
- **amssymb**: Símbolos matemáticos (para checkmarks)

## Solución de Problemas

### Error: "No file capitulo01_introduccion.tex"
Asegúrate de estar en el directorio `memoria/` donde están todos los archivos.

### Error: "Package tikz Error"
Verifica que todos los paquetes de TikZ estén instalados en MiKTeX.

### Warnings de fancyhdr
Son advertencias menores sobre el tamaño del encabezado. No afectan la compilación pero se pueden corregir ajustando `\headheight`.

## Mantenimiento

- Mantener todos los archivos de capítulo en el mismo directorio
- No usar caracteres especiales (acentos, ñ) en nombres de archivo
- Hacer backup antes de cambios importantes
- Compilar periódicamente para detectar errores temprano
