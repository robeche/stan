"""
📄 document_chunker.py - Sistema de Chunking para RAG

Este módulo proporciona funcionalidades para dividir documentos Markdown en fragmentos
optimizados para sistemas de Retrieval-Augmented Generation (RAG).

Características principales:
- División inteligente respetando la estructura del documento
- Fragmentos de tamaño configurable con solapamiento (overlap)
- Preservación del contexto con metadatos
- Múltiples estrategias de chunking disponibles
- Fácil de extender y personalizar

Autor: Sistema RAG
Fecha: Enero 2026
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ChunkingStrategy(Enum):
    """Estrategias de chunking disponibles"""
    FIXED_SIZE = "fixed_size"  # Tamaño fijo con overlap
    SEMANTIC = "semantic"  # Por secciones/párrafos
    HYBRID = "hybrid"  # Combinación de ambas


@dataclass
class ChunkConfig:
    """Configuración para el chunking de documentos"""
    chunk_size: int = 1000  # Tamaño objetivo en caracteres
    chunk_overlap: int = 200  # Solapamiento entre chunks en caracteres
    min_chunk_size: int = 100  # Tamaño mínimo de chunk
    max_chunk_size: int = 2000  # Tamaño máximo de chunk
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    preserve_tables: bool = True  # No dividir tablas
    preserve_code_blocks: bool = True  # No dividir bloques de código
    include_metadata: bool = True  # Incluir metadatos en chunks


@dataclass
class Chunk:
    """Representa un fragmento de documento"""
    content: str  # Contenido del chunk
    chunk_id: int  # ID único del chunk
    metadata: Dict[str, Any]  # Metadatos adicionales
    
    def __str__(self):
        return f"Chunk {self.chunk_id}: {len(self.content)} chars"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el chunk a diccionario"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': self.metadata,
            'length': len(self.content)
        }


class DocumentChunker:
    """
    Clase principal para chunking de documentos Markdown.
    
    Soporta múltiples estrategias de división y configuraciones personalizables.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Inicializa el chunker con la configuración especificada.
        
        Args:
            config: Configuración de chunking (usa defaults si no se proporciona)
        """
        self.config = config or ChunkConfig()
        self.chunks: List[Chunk] = []
        
    def chunk_document(self, file_path: str) -> List[Chunk]:
        """
        Divide un documento Markdown en chunks.
        
        Args:
            file_path: Ruta al archivo Markdown
            
        Returns:
            Lista de chunks generados
        """
        print(f"\n{'='*60}")
        print(f"📄 CHUNKING DE DOCUMENTO")
        print(f"{'='*60}")
        print(f"Archivo: {os.path.basename(file_path)}")
        print(f"Estrategia: {self.config.strategy.value}")
        print(f"Tamaño objetivo: {self.config.chunk_size} caracteres")
        print(f"Overlap: {self.config.chunk_overlap} caracteres")
        print(f"{'='*60}\n")
        
        # Leer documento
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Seleccionar estrategia
        if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            self.chunks = self._chunk_fixed_size(content)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            self.chunks = self._chunk_semantic(content)
        else:  # HYBRID
            self.chunks = self._chunk_hybrid(content)
        
        # Añadir metadatos globales
        for chunk in self.chunks:
            chunk.metadata['source_file'] = os.path.basename(file_path)
            chunk.metadata['total_chunks'] = len(self.chunks)
        
        self._print_statistics()
        return self.chunks
    
    def _chunk_fixed_size(self, content: str) -> List[Chunk]:
        """
        Divide el contenido en chunks de tamaño fijo con overlap.
        
        Args:
            content: Contenido completo del documento
            
        Returns:
            Lista de chunks
        """
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(content):
            # Calcular el final del chunk
            end = start + self.config.chunk_size
            
            # Si no es el último chunk, buscar un punto de corte natural
            if end < len(content):
                # Buscar el final de párrafo más cercano
                next_paragraph = content.find('\n\n', end - 100, end + 100)
                if next_paragraph != -1:
                    end = next_paragraph + 2
                else:
                    # Si no hay párrafo, buscar final de línea
                    next_line = content.find('\n', end - 50, end + 50)
                    if next_line != -1:
                        end = next_line + 1
            
            # Extraer contenido del chunk
            chunk_content = content[start:end].strip()
            
            if len(chunk_content) >= self.config.min_chunk_size:
                metadata = {
                    'start_pos': start,
                    'end_pos': end,
                    'strategy': 'fixed_size'
                }
                chunks.append(Chunk(chunk_content, chunk_id, metadata))
                chunk_id += 1
            
            # Mover al siguiente chunk con overlap
            start = end - self.config.chunk_overlap
        
        return chunks
    
    def _chunk_semantic(self, content: str) -> List[Chunk]:
        """
        Divide el contenido respetando la estructura semántica (secciones, páginas).
        
        Args:
            content: Contenido completo del documento
            
        Returns:
            Lista de chunks
        """
        chunks = []
        chunk_id = 0
        
        # Dividir por páginas si existen marcadores
        pages = re.split(r'\n## Página \d+\n', content)
        
        for page_idx, page_content in enumerate(pages):
            if not page_content.strip():
                continue
            
            # Dividir la página en secciones
            sections = self._extract_sections(page_content)
            
            for section in sections:
                # Si la sección es muy grande, dividirla
                if len(section['content']) > self.config.max_chunk_size:
                    # Dividir la sección grande en párrafos
                    sub_chunks = self._split_large_section(section['content'])
                    for sub_chunk in sub_chunks:
                        metadata = {
                            'page': page_idx + 1 if page_idx > 0 else None,
                            'section_title': section.get('title'),
                            'strategy': 'semantic'
                        }
                        chunks.append(Chunk(sub_chunk, chunk_id, metadata))
                        chunk_id += 1
                else:
                    # Sección completa como chunk
                    metadata = {
                        'page': page_idx + 1 if page_idx > 0 else None,
                        'section_title': section.get('title'),
                        'strategy': 'semantic'
                    }
                    chunks.append(Chunk(section['content'], chunk_id, metadata))
                    chunk_id += 1
        
        return chunks
    
    def _chunk_hybrid(self, content: str) -> List[Chunk]:
        """
        Estrategia híbrida: semántica primero, luego tamaño fijo si es necesario.
        
        Args:
            content: Contenido completo del documento
            
        Returns:
            Lista de chunks
        """
        chunks = []
        chunk_id = 0
        
        # Dividir por páginas si existen
        page_pattern = r'(## Página \d+)'
        parts = re.split(page_pattern, content)
        
        current_page = None
        i = 0
        
        while i < len(parts):
            # Detectar marcador de página
            if parts[i].startswith('## Página'):
                page_match = re.search(r'Página (\d+)', parts[i])
                current_page = int(page_match.group(1)) if page_match else None
                i += 1
                continue
            
            page_content = parts[i]
            if not page_content.strip():
                i += 1
                continue
            
            # Extraer secciones de la página
            sections = self._extract_sections(page_content)
            
            for section in sections:
                section_content = section['content']
                
                # Si la sección es pequeña, crear un chunk único
                if len(section_content) <= self.config.max_chunk_size:
                    metadata = {
                        'page': current_page,
                        'section_title': section.get('title'),
                        'strategy': 'hybrid_semantic'
                    }
                    chunks.append(Chunk(section_content, chunk_id, metadata))
                    chunk_id += 1
                else:
                    # Sección grande: dividir respetando párrafos
                    sub_chunks = self._split_large_section(
                        section_content,
                        preserve_structure=True
                    )
                    
                    for sub_chunk in sub_chunks:
                        metadata = {
                            'page': current_page,
                            'section_title': section.get('title'),
                            'strategy': 'hybrid_split'
                        }
                        chunks.append(Chunk(sub_chunk, chunk_id, metadata))
                        chunk_id += 1
            
            i += 1
        
        return chunks
    
    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Extrae secciones del contenido basándose en encabezados Markdown.
        
        Args:
            content: Contenido a dividir
            
        Returns:
            Lista de diccionarios con título y contenido de cada sección
        """
        sections = []
        
        # Buscar encabezados de nivel 2 o superior (##, ###, etc.)
        header_pattern = r'^(#{2,})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = {'title': None, 'content': ''}
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Guardar sección anterior si tiene contenido
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Iniciar nueva sección
                current_section = {
                    'title': header_match.group(2),
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # Guardar última sección
        if current_section['content'].strip():
            sections.append(current_section)
        
        # Si no se encontraron secciones, tratar todo el contenido como una sección
        if not sections:
            sections.append({'title': None, 'content': content})
        
        return sections
    
    def _split_large_section(self, content: str, preserve_structure: bool = True) -> List[str]:
        """
        Divide una sección grande en sub-chunks respetando la estructura.
        
        Args:
            content: Contenido de la sección
            preserve_structure: Si True, respeta párrafos y tablas
            
        Returns:
            Lista de sub-chunks
        """
        sub_chunks = []
        
        # Dividir por párrafos
        paragraphs = content.split('\n\n')
        
        current_chunk = ''
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Detectar tablas (no dividir)
            is_table = self._is_table(para)
            
            # Si añadir este párrafo excede el tamaño máximo
            if len(current_chunk) + len(para) > self.config.chunk_size and current_chunk:
                # Guardar chunk actual
                sub_chunks.append(current_chunk.strip())
                
                # Iniciar nuevo chunk con overlap si es posible
                if self.config.chunk_overlap > 0 and not is_table:
                    overlap_text = current_chunk[-self.config.chunk_overlap:].strip()
                    current_chunk = overlap_text + '\n\n' + para + '\n\n'
                else:
                    current_chunk = para + '\n\n'
            else:
                current_chunk += para + '\n\n'
        
        # Añadir último chunk
        if current_chunk.strip():
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks
    
    def _is_table(self, text: str) -> bool:
        """
        Detecta si un texto es una tabla Markdown.
        
        Args:
            text: Texto a verificar
            
        Returns:
            True si es una tabla
        """
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Buscar línea separadora de tabla (|---|---|)
        for line in lines[:3]:  # Verificar primeras 3 líneas
            if re.match(r'^\s*\|[\s\-:|]+\|\s*$', line):
                return True
        
        # También detectar tablas LaTeX
        if '\\begin{tabular}' in text or '\\end{tabular}' in text:
            return True
        
        return False
    
    def save_chunks(self, output_dir: str, format: str = 'txt') -> List[str]:
        """
        Guarda los chunks en archivos individuales.
        
        Args:
            output_dir: Directorio donde guardar los chunks
            format: Formato de salida ('txt', 'md', 'json')
            
        Returns:
            Lista de rutas de archivos guardados
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        for chunk in self.chunks:
            if format == 'json':
                import json
                file_path = os.path.join(output_dir, f'chunk_{chunk.chunk_id:04d}.json')
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk.to_dict(), f, ensure_ascii=False, indent=2)
            else:
                ext = 'md' if format == 'md' else 'txt'
                file_path = os.path.join(output_dir, f'chunk_{chunk.chunk_id:04d}.{ext}')
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Escribir metadatos como comentario
                    f.write(f"<!-- Chunk ID: {chunk.chunk_id} -->\n")
                    f.write(f"<!-- Metadata: {chunk.metadata} -->\n\n")
                    f.write(chunk.content)
            
            saved_files.append(file_path)
        
        print(f"\n✓ {len(saved_files)} chunks guardados en: {output_dir}")
        return saved_files
    
    def save_chunks_combined(self, output_file: str) -> str:
        """
        Guarda todos los chunks en un único archivo con separadores.
        
        Args:
            output_file: Ruta del archivo de salida
            
        Returns:
            Ruta del archivo guardado
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Documento Dividido en Chunks para RAG\n\n")
            f.write(f"Total de chunks: {len(self.chunks)}\n")
            f.write(f"Configuración: {self.config}\n\n")
            f.write("="*80 + "\n\n")
            
            for chunk in self.chunks:
                f.write(f"## CHUNK {chunk.chunk_id}\n\n")
                f.write(f"**Metadatos:** {chunk.metadata}\n\n")
                f.write(f"**Longitud:** {len(chunk.content)} caracteres\n\n")
                f.write("---\n\n")
                f.write(chunk.content)
                f.write("\n\n" + "="*80 + "\n\n")
        
        print(f"✓ Chunks combinados guardados en: {output_file}")
        return output_file
    
    def _print_statistics(self):
        """Imprime estadísticas sobre los chunks generados"""
        if not self.chunks:
            return
        
        lengths = [len(chunk.content) for chunk in self.chunks]
        
        print(f"\n{'='*60}")
        print(f"📊 ESTADÍSTICAS DE CHUNKING")
        print(f"{'='*60}")
        print(f"Total de chunks: {len(self.chunks)}")
        print(f"Longitud promedio: {sum(lengths) / len(lengths):.0f} caracteres")
        print(f"Longitud mínima: {min(lengths)} caracteres")
        print(f"Longitud máxima: {max(lengths)} caracteres")
        print(f"Longitud total: {sum(lengths):,} caracteres")
        
        # Distribución por estrategia
        strategies = {}
        for chunk in self.chunks:
            strategy = chunk.metadata.get('strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        print(f"\nDistribución por estrategia:")
        for strategy, count in strategies.items():
            print(f"  - {strategy}: {count} chunks ({count/len(self.chunks)*100:.1f}%)")
        
        print(f"{'='*60}\n")


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def create_chunks_from_file(
    input_file: str,
    output_dir: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    strategy: str = 'hybrid'
) -> List[Chunk]:
    """
    Función de conveniencia para crear chunks desde un archivo.
    
    Args:
        input_file: Archivo Markdown de entrada
        output_dir: Directorio de salida para los chunks
        chunk_size: Tamaño objetivo de cada chunk
        overlap: Solapamiento entre chunks
        strategy: Estrategia de chunking ('fixed_size', 'semantic', 'hybrid')
        
    Returns:
        Lista de chunks generados
    """
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        strategy=ChunkingStrategy(strategy)
    )
    
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(input_file)
    chunker.save_chunks(output_dir, format='md')
    chunker.save_chunks_combined(os.path.join(output_dir, 'chunks_combined.md'))
    
    return chunks


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    # Ejemplo: chunking del documento concatenado
    import sys
    
    # Configuración por defecto
    INPUT_FILE = "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    OUTPUT_DIR = "output_simple/NREL5MW_Reduced/chunks"
    
    # Permitir argumentos desde línea de comandos
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    
    print("\n" + "="*80)
    print("🚀 SISTEMA DE CHUNKING PARA RAG")
    print("="*80)
    
    # Configuración personalizada
    config = ChunkConfig(
        chunk_size=1200,        # Fragmentos de ~1200 caracteres
        chunk_overlap=200,      # 200 caracteres de solapamiento
        min_chunk_size=300,     # Mínimo 300 caracteres
        max_chunk_size=2500,    # Máximo 2500 caracteres
        strategy=ChunkingStrategy.HYBRID,
        preserve_tables=True,
        include_metadata=True
    )
    
    # Crear chunker
    chunker = DocumentChunker(config)
    
    # Procesar documento
    try:
        chunks = chunker.chunk_document(INPUT_FILE)
        
        # Guardar resultados
        chunker.save_chunks(OUTPUT_DIR, format='md')
        chunker.save_chunks(OUTPUT_DIR + '_json', format='json')
        chunker.save_chunks_combined(
            os.path.join(OUTPUT_DIR, 'documento_chunkeado.md')
        )
        
        print("\n✅ Chunking completado exitosamente!")
        print(f"📁 Chunks guardados en: {OUTPUT_DIR}")
        
    except FileNotFoundError:
        print(f"\n❌ Error: No se encontró el archivo: {INPUT_FILE}")
        print("Por favor, verifica la ruta del archivo.")
    except Exception as e:
        print(f"\n❌ Error durante el chunking: {e}")
        import traceback
        traceback.print_exc()
