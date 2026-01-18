import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor
import os
from pathlib import Path
import fitz  # PyMuPDF
import argparse
import sys
import re
import glob

# Load model and processor
model_id = "nvidia/NVIDIA-Nemotron-Parse-v1.1"
token = os.getenv("HF_TOKEN")

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"✓ Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Usando CPU")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token, use_fast=False)
model = AutoModel.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map=device,
    trust_remote_code=True,
    token=token
)

def parse_document(image_path):
    """Parse a single image with the model"""
    image = Image.open(image_path).convert("RGB")
    print(f"  Imagen cargada: {image.size}")
    
    task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"
    
    inputs = processor(images=[image], text=task_prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    print(f"  Generando texto...")
    
    from transformers import GenerationConfig
    generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
    generation_config.use_cache = False
    generation_config.max_new_tokens = 16384
    
    generated_ids = model.generate(**inputs, generation_config=generation_config)
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"  Tokens generados: {generated_ids.shape[1]}")
    print(f"  Longitud resultado: {len(result)} caracteres")
    
    if not result.strip().endswith('>'):
        print("  ⚠ Advertencia: El output parece estar truncado")

    # Si la respuesta no empieza con coordenadas, añadir <x_0><y_0>
    if not re.match(r'^\s*<x_[\d.]+><y_[\d.]+>', result):
        result = f"<x_0><y_0>{result}"
    
    return result


def draw_selected_boxes(image, parsed_text, output_path):
    """Draw bounding boxes only for Table, Picture/Figure, and Formula with labels."""
    allowed = {"Table": "#FF5722", "Figure": "#9C27B0", "Picture": "#9C27B0", "Formula": "#FF9800"}
    pattern = r'<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>'
    matches = list(re.finditer(pattern, parsed_text, re.DOTALL))
    # Ordenar por coordenada Y luego X para consistencia
    matches = sorted(matches, key=lambda m: (float(m.group(2)), float(m.group(1))))

    # Ordenar por coordenada Y luego X para consistencia
    matches = sorted(matches, key=lambda m: (float(m.group(2)), float(m.group(1))))

    if not matches:
        return 0

    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)

    # Fuente para etiquetas
    try:
        font_size = max(12, int(img_h * 0.015))
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    count = 0
    for m in matches:
        cls = m.group(6).strip()
        if cls not in allowed:
            continue
        x1 = float(m.group(1)); y1 = float(m.group(2)); x2 = float(m.group(4)); y2 = float(m.group(5))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        x1_px = int(x1 * img_w); y1_px = int(y1 * img_h)
        x2_px = int(x2 * img_w); y2_px = int(y2 * img_h)

        # Padding adicional por clase (en píxeles)
        if cls in ("Figure", "Picture"):
            pad_x = 15
            pad_y = 0
        elif cls == "Table":
            pad_x = 25
            pad_y = 0
        elif cls == "Formula":
            pad_x = 30
            pad_y = 0
        else:
            pad = 0

        x1_px = max(0, x1_px - pad_x)
        y1_px = max(0, y1_px - pad_y)
        x2_px = min(img_w, x2_px + pad_x)
        y2_px = min(img_h, y2_px + pad_y)

        draw.rectangle([(x1_px, y1_px), (x2_px, y2_px)], outline=allowed[cls], width=3)

        # Etiqueta con la clase
        label = cls
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        except:
            tw = len(label) * 8; th = 12

        pad = 2
        lx = x1_px
        ly = y1_px - th - 4
        if ly < 0:
            ly = y1_px + 2
        draw.rectangle([(lx - pad, ly - pad), (lx + tw + pad, ly + th + pad)], fill=allowed[cls])
        draw.text((lx, ly), label, fill="white", font=font)

        count += 1

    # Guardar siempre la imagen (aunque no haya boxes)
    image.save(output_path)
    return count


def _slugify(text: str, fallback: str) -> str:
    # Genera un nombre de archivo seguro y corto
    cleaned = re.sub(r'\s+', '_', text.strip())
    cleaned = re.sub(r'[^A-Za-z0-9._-]', '', cleaned)
    if not cleaned:
        cleaned = fallback
    # Limitar longitud para evitar rutas largas
    return cleaned[:80]


def _slugify_full(text: str, fallback: str) -> str:
    # Genera un nombre de archivo usando el caption completo (solo sanea caracteres)
    cleaned = re.sub(r'\s+', '_', text.strip())
    cleaned = re.sub(r'[^A-Za-z0-9._-]', '', cleaned)
    if not cleaned:
        cleaned = fallback
    # No truncar; confiar en el caption completo
    return cleaned


def _sanitize_doc_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip())


def reorder_by_y(parsed_text: str) -> str:
    """Reordena todos los elementos por coordenada Y y luego X."""
    pattern = r'<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>'
    matches = list(re.finditer(pattern, parsed_text, re.DOTALL))
    if not matches:
        return parsed_text
    matches = sorted(matches, key=lambda m: (float(m.group(2)), float(m.group(1))))
    return "\n\n".join(m.group(0) for m in matches)


def replace_picture_content(parsed_text: str, doc_name: str, page_num: int) -> str:
    """Reemplaza el contenido de Picture/Figure por el nombre de archivo doc_p{page}_fig{n}."""
    pattern = r'<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>'
    matches = list(re.finditer(pattern, parsed_text, re.DOTALL))
    if not matches:
        return parsed_text

    # Orden consistente por Y, luego X
    sorted_matches = sorted(matches, key=lambda m: (float(m.group(2)), float(m.group(1))))

    out = []
    last_idx = 0
    fig_idx = 1
    for m in sorted_matches:
        cls = m.group(6).strip()
        if cls not in ("Picture", "Figure"):
            continue
        filename = f"{doc_name}_p{page_num}_fig{fig_idx}.png"
        fig_idx += 1
        start, end = m.span(3)  # contenido
        out.append(parsed_text[last_idx:start])
        out.append(filename)
        last_idx = end
    out.append(parsed_text[last_idx:])
    return "".join(out)


def combine_raw_to_markdown(output_dir: Path, doc_name: str):
    """Concatena todos los raw en un markdown, limpiando coords/tags y renderizando figuras con caption/filename."""
    raw_dir = output_dir / "raw_output"
    md_lines = []
    md_lines.append(f"# {doc_name}\n\n")

    pattern = r'<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>'

    files = sorted(raw_dir.glob("page_*_raw.txt"), key=lambda p: int(re.search(r'page_(\d+)_raw', p.name).group(1)))

    for fpath in files:
        mpage = re.search(r'page_(\d+)_raw', fpath.name)
        page_num = int(mpage.group(1)) if mpage else 0
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
        text = reorder_by_y(text)
        matches = list(re.finditer(pattern, text, re.DOTALL))
        matches = sorted(matches, key=lambda m: (float(m.group(2)), float(m.group(1))))

        md_lines.append(f"\n\n## Página {page_num}\n\n")

        pending_fig = None

        for m in matches:
            cls = m.group(6).strip()
            content = m.group(3).strip()

            if cls in ("Picture", "Figure"):
                pending_fig = content  # filename ya insertado
                continue

            if cls == "Caption" and pending_fig:
                caption = content
                md_lines.append(f"![{caption}](figures/{pending_fig})\n\n")
                pending_fig = None
                continue

            if cls == "Table":
                md_lines.append(f"{content}\n\n")
            elif cls == "List-item":
                md_lines.append(f"- {content}\n")
            elif cls in ("Title", "Section-header"):
                md_lines.append(f"## {content}\n\n")
            elif cls == "Text":
                md_lines.append(f"{content}\n\n")
            elif cls == "Footnote":
                md_lines.append(f"_{content}_\n\n")
            elif cls == "Formula":
                md_lines.append(f"{content}\n\n")
            elif cls == "Page-footer":
                # opcional: omitir pies de página
                continue
            else:
                # otros: solo texto
                md_lines.append(f"{content}\n\n")

        # Si quedó una figura sin caption, incluirla sin título
        if pending_fig:
            md_lines.append(f"![Figura](figures/{pending_fig})\n\n")

    out_path = output_dir / "documento_concatenado.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("".join(md_lines))
    print(f"✓ Markdown combinado guardado en: {out_path}")


def extract_assets(image, parsed_text, output_dir, doc_name: str, page_num: int):
    """Extrae figuras y tablas a PNG, usando el caption más cercano como nombre."""
    out_fig = Path(output_dir) / "figures"
    out_tab = Path(output_dir) / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    pattern = r'<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>'
    matches = list(re.finditer(pattern, parsed_text, re.DOTALL))
    if not matches:
        return 0, 0

    img_w, img_h = image.size
    # Recolectar captions
    captions = []
    for m in matches:
        cls = m.group(6).strip()
        if cls == "Caption":
            content = m.group(3).strip()
            y_mid = (float(m.group(2)) + float(m.group(5))) / 2.0
            captions.append({"y": y_mid, "text": content})

    fig_count = 0
    tab_count = 0

    fig_per_page = 0
    for m in matches:
        cls = m.group(6).strip()
        if cls not in ("Table", "Figure", "Picture"):
            continue

        x1 = float(m.group(1)); y1 = float(m.group(2)); x2 = float(m.group(4)); y2 = float(m.group(5))
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Padding por clase (alineado con las anotaciones)
        if cls == "Table":
            pad_x, pad_y = 25, 0
        else:  # Figure/Picture
            pad_x, pad_y = 15, 0

        x1_px = max(0, int(x1 * img_w) - pad_x)
        y1_px = max(0, int(y1 * img_h) - pad_y)
        x2_px = min(img_w, int(x2 * img_w) + pad_x)
        y2_px = min(img_h, int(y2 * img_h) + pad_y)

        crop = image.crop((x1_px, y1_px, x2_px, y2_px))

        # Buscar caption más cercano por debajo (o global más cercano)
        y_mid_elem = (y1 + y2) / 2.0
        caption_text = ""
        below = [c for c in captions if c["y"] >= y_mid_elem]
        nearest = None
        if below:
            nearest = min(below, key=lambda c: c["y"] - y_mid_elem)
        elif captions:
            nearest = min(captions, key=lambda c: abs(c["y"] - y_mid_elem))

        if nearest:
            caption_text = nearest["text"]

        if cls == "Table":
            tab_count += 1
            filename = _slugify_full(caption_text, f"table_{tab_count}") + ".png"
            crop.save(out_tab / filename)
        else:
            fig_count += 1
            fig_per_page += 1
            filename = f"{doc_name}_p{page_num}_fig{fig_per_page}.png"
            crop.save(out_fig / filename)

    return fig_count, tab_count

def pdf_to_images(pdf_path, dpi=300):
    """Convert PDF to images (one per page)"""
    print(f"\nConvirtiendo PDF a imágenes (DPI: {dpi})...")
    pdf_document = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append({
            'image': img,
            'page_num': page_num + 1
        })
        print(f"  Página {page_num + 1}/{len(pdf_document)} convertida ({img.size[0]}x{img.size[1]} px)")
    
    pdf_document.close()
    print(f"✓ {len(images)} páginas convertidas\n")
    return images

def process_pdf(pdf_path, output_directory, dpi=300):
    """Process PDF: convert to images and parse each page"""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pages").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw_output").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotated_pages").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    
    doc_name = _sanitize_doc_name(Path(pdf_path).stem)

    # Convert PDF to images
    page_images = pdf_to_images(pdf_path, dpi=dpi)
    total_pages = len(page_images)
    
    # Process each page
    for idx, page_data in enumerate(page_images, 1):
        page_num = page_data['page_num']
        image = page_data['image']
        
        print(f"{'='*80}")
        print(f"PROCESANDO PÁGINA {page_num} de {total_pages} ({idx}/{total_pages})")
        print('='*80)
        
        # Save page image
        page_path = output_dir / "pages" / f"page_{page_num}.png"
        image.save(page_path)
        print(f"✓ Página guardada: {page_path.name}")
        
        # Parse page (guardar PNG para evitar compresión)
        temp_image_path = output_dir / f"temp_page_{page_num}.png"
        image.save(temp_image_path)
        
        result = parse_document(str(temp_image_path))
        # Reordenar por Y y reemplazar contenido de figuras por filename
        result = reorder_by_y(result)
        result = replace_picture_content(result, doc_name, page_num)
        
        # Save raw output
        raw_output_path = output_dir / "raw_output" / f"page_{page_num}_raw.txt"
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"✓ Salida raw guardada: {raw_output_path.name}")

        # Draw bounding boxes only for tables, figures/images, formulas
        annotated_path = output_dir / "annotated_pages" / f"page_{page_num}_annotated.png"
        boxes = draw_selected_boxes(image.copy(), result, annotated_path)
        if boxes:
            print(f"✓ Bounding boxes dibujados: {boxes} -> {annotated_path.name}")
        else:
            print("- Sin bounding boxes seleccionados en esta página")

        # Extraer assets (figuras y tablas)
        figs, tabs = extract_assets(image, result, output_dir, doc_name, page_num)
        print(f"✓ Assets extraídos: figuras={figs}, tablas={tabs}")
        
        # Clean up temp file
        temp_image_path.unlink()
        
        print(f"✓ Página {page_num} completada")
        print(f"📊 Progreso: {idx}/{total_pages} páginas ({int(idx/total_pages*100)}%)\n")
    
    print(f"{'='*80}")
    print(f"✓ PROCESO COMPLETADO")
    print(f"{'='*80}")
    print(f"✓ Páginas guardadas en: {output_dir}/pages/")
    print(f"✓ Salidas raw en: {output_dir}/raw_output/")

    # Combinar todos los raw en un solo markdown
    combine_raw_to_markdown(output_dir, doc_name)

def process_image(image_path, output_directory):
    """Process a single image"""
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "pages").mkdir(exist_ok=True)
    (output_dir / "raw_output").mkdir(exist_ok=True)
    (output_dir / "annotated_pages").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    
    print(f"{'='*80}")
    print(f"PROCESANDO IMAGEN: {Path(image_path).name}")
    print('='*80)
    
    # Save copy of image
    image = Image.open(image_path).convert("RGB")
    page_path = output_dir / "pages" / "page_1.png"
    image.save(page_path)
    print(f"✓ Imagen guardada: {page_path.name}")
    
    # Parse image sin padding
    temp_image_path = output_dir / "temp_page_1.png"
    image.save(temp_image_path)

    result = parse_document(str(temp_image_path))
    doc_name = _sanitize_doc_name(Path(image_path).stem)
    result = reorder_by_y(result)
    result = replace_picture_content(result, doc_name, 1)
    
    # Save raw output
    raw_output_path = output_dir / "raw_output" / "page_1_raw.txt"
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"✓ Salida raw guardada: {raw_output_path.name}")

    # Draw bounding boxes only for tables, figures/images, formulas
    annotated_path = output_dir / "annotated_pages" / "page_1_annotated.png"
    boxes = draw_selected_boxes(image.copy(), result, annotated_path)
    if boxes:
        print(f"✓ Bounding boxes dibujados: {boxes} -> {annotated_path.name}")
    else:
        print("- Sin bounding boxes seleccionados en esta imagen")

    # Extraer assets (figuras y tablas)
    figs, tabs = extract_assets(image, result, output_dir, doc_name, 1)
    print(f"✓ Assets extraídos: figuras={figs}, tablas={tabs}")
    
    print(f"\n{'='*80}")
    print(f"✓ PROCESO COMPLETADO")
    print(f"{'='*80}")
    print(f"✓ Imagen en: {output_dir}/pages/")
    print(f"✓ Salida raw en: {output_dir}/raw_output/")
    combine_raw_to_markdown(output_dir, doc_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Parsea documentos PDF o imágenes usando NVIDIA Nemotron Parse (versión simplificada)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python parse_simple.py documento.pdf
  python parse_simple.py imagen.jpg -o salida_custom
  python parse_simple.py documento.pdf --dpi 200
        """
    )
    
    parser.add_argument('input_file', 
                        help='Archivo de entrada (PDF, JPG, PNG, etc.)')
    parser.add_argument('-o', '--output', 
                        default='output_simple',
                        help='Directorio de salida (por defecto: output_simple)')
    parser.add_argument('--dpi', 
                        type=int, 
                        default=300,
                        help='DPI para conversión de PDF (por defecto: 300)')
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.input_file).exists():
        print(f"❌ Error: El archivo '{args.input_file}' no existe")
        sys.exit(1)
    
    input_file = args.input_file
    dpi = args.dpi
    
    # Create output directory based on document name
    file_path = Path(input_file)
    document_name = file_path.stem
    output_directory = Path(args.output) / document_name
    
    print(f"📁 Directorio de salida: {output_directory}\n")

    # Detect if PDF or image
    is_pdf = file_path.suffix.lower() == '.pdf'

    if is_pdf:
        process_pdf(input_file, output_directory, dpi=dpi)
    else:
        process_image(input_file, output_directory)
