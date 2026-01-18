"""
Document chunker wrapper for Django integration.
"""
import sys
from pathlib import Path

# Add parent directory to path
PARENT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from document_chunker import DocumentChunker as OriginalChunker, ChunkConfig, ChunkingStrategy


class DocumentChunker:
    """Wrapper for DocumentChunker with Django-friendly interface."""
    
    def __init__(self, strategy='hybrid_semantic', chunk_size=512, overlap=50):
        """
        Initialize chunker.
        
        Args:
            strategy: Chunking strategy ('fixed_size', 'semantic', 'hybrid')
            chunk_size: Target size for chunks
            overlap: Overlap between chunks
        """
        # Map strategy names
        strategy_map = {
            'fixed_size': ChunkingStrategy.FIXED_SIZE,
            'fixed': ChunkingStrategy.FIXED_SIZE,
            'semantic': ChunkingStrategy.SEMANTIC,
            'hybrid': ChunkingStrategy.HYBRID,
            'hybrid_semantic': ChunkingStrategy.HYBRID,
        }
        
        strategy_enum = strategy_map.get(strategy, ChunkingStrategy.HYBRID)
        
        # Create config with settings to prevent paragraph cutting
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_chunk_size=200,  # Higher minimum to keep complete thoughts
            max_chunk_size=chunk_size * 4,  # Allow much larger chunks to preserve complete paragraphs
            strategy=strategy_enum,
            preserve_tables=True,  # Don't split tables
            preserve_code_blocks=True,  # Don't split code blocks
            include_metadata=True
        )
        
        self.chunker = OriginalChunker(config=config)
    
    def chunk_document(self, text):
        """
        Chunk a document text.
        
        Args:
            text: Document text to chunk
        
        Returns:
            list: List of chunk dictionaries with text and metadata
        """
        # Save text to temporary file since original chunker expects file path
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.md', delete=False) as f:
            f.write(text)
            temp_path = f.name
        
        try:
            # Use original chunker with file path
            chunks = self.chunker.chunk_document(temp_path)
            
            # Convert Chunk objects to dictionaries
            return [
                {
                    'text': chunk.content,
                    'chunk_id': chunk.chunk_id,
                    'metadata': chunk.metadata
                }
                for chunk in chunks
            ]
        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def chunk_file(self, file_path):
        """
        Chunk a text file.
        
        Args:
            file_path: Path to text file
        
        Returns:
            list: List of chunk dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.chunk_document(text)
