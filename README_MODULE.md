# Nemotron Parser

Módulo Python para parsear documentos PDF e imágenes usando NVIDIA Nemotron Parse.

## Instalación

```bash
pip install -r requirements.txt
```

Asegúrate de tener configurado tu token de Hugging Face:

```bash
export HF_TOKEN="tu_token_aqui"
```

## Uso como Módulo

### Ejemplo básico

```python
from nemotron_parser import NemotronParser

# Inicializar parser
parser = NemotronParser(dpi=300, verbose=True)

# Procesar PDF
result = parser.process_pdf("documento.pdf", "output_dir")

# Crear documento Markdown
parser.create_markdown_document(
    result['parsed_results'],
    result['elements'],
    "output_dir/documento.md"
)
```

### Procesar imagen individual

```python
from nemotron_parser import NemotronParser

parser = NemotronParser()
result = parser.process_image("imagen.jpg", "output_dir")

print(f"Elementos extraídos: {len(result['elements'])}")
```

### Procesamiento en lote

```python
from nemotron_parser import NemotronParser
from pathlib import Path

parser = NemotronParser(dpi=200)  # DPI más bajo = más rápido

for pdf_file in Path("documentos/").glob("*.pdf"):
    output_dir = f"output/{pdf_file.stem}"
    result = parser.process_pdf(str(pdf_file), output_dir)
    print(f"✓ {pdf_file.name} procesado")
```

### Acceso a elementos específicos

```python
from nemotron_parser import NemotronParser

parser = NemotronParser()
result = parser.process_pdf("documento.pdf", "output")

# Filtrar tablas
tables = [e for e in result['elements'] if e['type'] == 'Table']
print(f"Tablas encontradas: {len(tables)}")

# Filtrar figuras
figures = [e for e in result['elements'] if e['type'] in ['Picture', 'Figure']]
for fig in figures:
    print(f"Figura en página {fig['page']}: {fig['reference']}")
```

## Uso como Script CLI

El script original `parse_local.py` sigue funcionando:

```bash
# Procesar PDF
python parse_local.py documento.pdf

# Especificar directorio de salida
python parse_local.py documento.pdf -o mi_salida

# Cambiar resolución (DPI)
python parse_local.py documento.pdf --dpi 200

# Procesar imagen
python parse_local.py imagen.jpg -o salida_imagen
```

## Parámetros de NemotronParser

### Constructor

- **dpi** (int, default=300): Resolución para conversión de PDFs
  - 150-200: Rápido, menor calidad
  - 300: Balance óptimo (recomendado)
  - 400-600: Alta calidad, más lento

- **verbose** (bool, default=True): Mostrar mensajes de progreso

- **token** (str, optional): Token de Hugging Face (si None, usa HF_TOKEN del entorno)

- **model_id** (str, default="nvidia/NVIDIA-Nemotron-Parse-v1.1"): ID del modelo

## Métodos Principales

### `process_pdf(pdf_path, output_directory)`

Procesa un PDF completo página por página.

**Returns:**
```python
{
    'parsed_results': [{'page': 1, 'result': '...'}, ...],
    'elements': [...],
    'output_dir': Path('...')
}
```

### `process_image(image_path, output_directory)`

Procesa una imagen individual.

**Returns:**
```python
{
    'parsed_result': '...',
    'elements': [...],
    'output_dir': Path('...')
}
```

### `create_markdown_document(parsed_results, elements, output_path)`

Crea un documento Markdown combinado con todos los elementos.

## Estructura de Elementos

Cada elemento extraído tiene:

```python
{
    'type': 'Table' | 'Picture' | 'Figure' | 'Text' | 'Title' | ...,
    'content': '...',
    'bbox': (x1, y1, x2, y2),  # Coordenadas en píxeles
    'coords_normalized': (x1, y1, x2, y2),  # Coordenadas normalizadas
    'page': 1,  # Número de página (solo PDFs)
    'reference': '[Table 1](tables/table_1.png)',  # Para tablas/figuras
    'image_path': 'output/tables/table_1.png',  # Para tablas/figuras
    'markdown_path': 'output/tables/table_1.md',  # Para tablas
    'details_path': 'output/images/figure_1.md'  # Para figuras (con reprocesamiento)
}
```

## Tipos de Elementos Detectados

- **Table**: Tablas
- **Picture/Figure**: Imágenes y figuras
- **Text**: Texto normal
- **Title**: Títulos
- **Section-header**: Encabezados de sección
- **Caption**: Leyendas
- **List-item**: Items de lista
- **Formula**: Fórmulas matemáticas
- **Footnote**: Notas al pie
- **Page-header/Page-footer**: Encabezados y pies de página
- **Bibliography**: Referencias bibliográficas
- **TOC**: Tabla de contenidos

## Archivos Generados

```
output_dir/
├── documento_parseado.md       # Documento Markdown combinado
├── raw_output/                 # Salidas crudas del modelo
│   ├── page_1_raw.txt
│   ├── page_2_raw.txt
│   └── ...
├── tables/                     # Tablas extraídas
│   ├── table_1.png            # Imagen de la tabla
│   ├── table_1.md             # Tabla en Markdown
│   └── ...
└── images/                     # Figuras extraídas
    ├── figure_1.png           # Imagen de la figura
    ├── figure_1.md            # Detalles extraídos
    └── ...
```

## Ejemplos Completos

Ver [example_usage.py](example_usage.py) para ejemplos detallados.

## Requisitos

- Python 3.8+
- CUDA compatible GPU (recomendado)
- Token de Hugging Face con acceso al modelo NVIDIA-Nemotron-Parse-v1.1

## Optimización de Velocidad

### Reducir DPI
```python
parser = NemotronParser(dpi=200)  # ~30-50% más rápido que 300 DPI
```

### Desactivar reprocesamiento de figuras
Editar `nemotron_parser.py` y comentar la llamada a `reprocess_figure()` en el método `extract_elements()`.

### Procesamiento en batch
Reutilizar la misma instancia de `NemotronParser` para múltiples documentos evita recargar el modelo.

## Notas

- El reprocesamiento de figuras duplica el tiempo de procesamiento pero mejora significativamente la extracción de títulos, leyendas y texto en imágenes
- Los márgenes de extracción son más generosos para figuras (200px) que para tablas (120px) para capturar contexto completo
- El autocrop elimina bordes blancos automáticamente
