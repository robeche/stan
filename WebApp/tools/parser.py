"""
Parser wrapper for Nemotron document processing.
Adapts parse_local.py functions for Django integration.
"""
import sys
from pathlib import Path

# Add parent directory to path to access original modules
PARENT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

# Import original functions
from parse_local import (
    process_pdf,
    process_image,
    combine_raw_to_markdown
)


def parse_document(input_path, output_dir, dpi=300):
    """
    Parse a document (PDF or image) using Nemotron.
    
    Args:
        input_path: Path to PDF or image file
        output_dir: Directory for output files
        dpi: DPI for PDF rendering (default: 300)
    
    Returns:
        dict: Processing results with paths and statistics
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine file type and process accordingly
    if input_path.suffix.lower() == '.pdf':
        result = process_pdf(str(input_path), str(output_dir), dpi=dpi)
    elif input_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        result = process_image(str(input_path), str(output_dir))
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Generate combined markdown
    doc_name = input_path.stem
    combine_raw_to_markdown(output_dir, doc_name)
    
    # Prepare result dictionary
    return {
        'success': True,
        'output_dir': str(output_dir),
        'doc_name': doc_name,
        'pages_dir': str(output_dir / 'pages'),
        'annotated_dir': str(output_dir / 'annotated_pages'),
        'figures_dir': str(output_dir / 'figures'),
        'tables_dir': str(output_dir / 'tables'),
        'concatenated_file': str(output_dir / 'documento_concatenado.md'),
    }
