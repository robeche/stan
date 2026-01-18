"""
Embedding generator wrapper for Django integration.
"""
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
PARENT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from embedding_generator import EmbeddingGenerator as OriginalGenerator


class EmbeddingGenerator:
    """Wrapper for EmbeddingGenerator with Django-friendly interface."""
    
    def __init__(self, model_name='BAAI/bge-m3', device='cuda'):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cuda' or 'cpu')
        """
        self.generator = OriginalGenerator(model_name=model_name, device=device)
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings or single text string
        
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        return self.generator.generate_embedding(texts)
    
    def generate_single_embedding(self, text):
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string
        
        Returns:
            numpy.ndarray: Single embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0]
