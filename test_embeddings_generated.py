"""
Script de prueba rápida de embeddings generados
"""

import json
import numpy as np
from pathlib import Path


def test_embeddings():
    """Prueba rápida de los embeddings generados."""
    print("="*60)
    print("PRUEBA DE EMBEDDINGS GENERADOS")
    print("="*60)
    
    # Cargar metadata
    metadata_path = Path("output_rag/embeddings/embeddings_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"\n📊 Información General:")
    print(f"  Modelo: {metadata['model']}")
    print(f"  Dimensiones: {metadata['embedding_dimension']}")
    print(f"  Chunks procesados: {metadata['num_chunks']}")
    print(f"  Tiempo generación: {metadata['generation_time_seconds']:.3f}s")
    print(f"  Normalizados: {metadata['normalized']}")
    print(f"  Fecha: {metadata['timestamp']}")
    
    # Cargar embeddings como numpy array
    embeddings_path = Path("output_rag/embeddings/embeddings.npy")
    embeddings = np.load(embeddings_path)
    
    print(f"\n📦 Array de Embeddings:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Tipo: {embeddings.dtype}")
    print(f"  Tamaño en memoria: {embeddings.nbytes / 1024:.2f} KB")
    
    # Cargar un chunk de ejemplo
    chunk_path = Path("output_rag/embeddings/chunk_0005.json")
    with open(chunk_path, 'r', encoding='utf-8') as f:
        chunk = json.load(f)
    
    print(f"\n📄 Ejemplo de Chunk (chunk_0005):")
    print(f"  Contenido: {chunk['content'][:100]}...")
    print(f"  Length: {chunk['length']}")
    print(f"  Embedding shape: {len(chunk['embedding'])}")
    print(f"  Primeros 5 valores: {chunk['embedding'][:5]}")
    
    # Calcular similitud entre chunks
    print(f"\n🔍 Similitud entre chunks (cosine similarity):")
    for i in range(0, min(5, len(embeddings))):
        for j in range(i+1, min(5, len(embeddings))):
            similarity = np.dot(embeddings[i], embeddings[j])
            print(f"  Chunk {i} vs Chunk {j}: {similarity:.4f}")
    
    # Encontrar chunks más similares
    print(f"\n🎯 Top 3 chunks más similares al chunk 5:")
    chunk_5_emb = embeddings[5]
    similarities = embeddings @ chunk_5_emb  # Producto punto (ya normalizados)
    top_indices = np.argsort(similarities)[::-1][1:4]  # Excluir el mismo
    
    for idx in top_indices:
        print(f"  Chunk {idx}: similarity = {similarities[idx]:.4f}")
    
    print(f"\n" + "="*60)
    print("✓ Embeddings listos para usar en sistema RAG!")
    print("="*60)


if __name__ == "__main__":
    test_embeddings()
