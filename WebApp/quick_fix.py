"""
Quick fix for ChromaDB corruption - checks status and rebuilds if needed
"""
import os
import django
import shutil
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

from django.conf import settings
from admin_panel.models import Document, Chunk
from tools.vector_store import VectorStore

print("=" * 80)
print("DIAGNÓSTICO Y REPARACIÓN DE CHROMADB")
print("=" * 80)

# Check Django status
total_docs = Document.objects.count()
total_chunks = Chunk.objects.count()
chunks_with_embeddings = Chunk.objects.exclude(embedding_vector__isnull=True).exclude(embedding_vector='').count()

print(f"\n📊 ESTADO DJANGO:")
print(f"  Documentos: {total_docs}")
print(f"  Chunks totales: {total_chunks}")
print(f"  Chunks con embeddings: {chunks_with_embeddings}")

# Check if we have data to rebuild
if chunks_with_embeddings == 0:
    print("\n❌ No hay chunks con embeddings en Django")
    print("   Solución: Procesa un documento nuevo desde la interfaz web")
    exit(1)

# Try to access ChromaDB
print(f"\n🔍 VERIFICANDO CHROMADB...")
chroma_dir = Path(settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY'])
print(f"  Directorio: {chroma_dir}")
print(f"  Existe: {chroma_dir.exists()}")

# Delete corrupted ChromaDB
if chroma_dir.exists():
    print(f"\n🗑️  Eliminando ChromaDB corrupto...")
    shutil.rmtree(chroma_dir)
    print(f"  ✓ Eliminado")

# Mark all chunks as not indexed
print(f"\n📝 Actualizando estado en Django...")
Chunk.objects.all().update(indexed_in_chromadb=False, chromadb_id=None)
print(f"  ✓ {total_chunks} chunks marcados como no indexados")

# Rebuild ChromaDB
print(f"\n🔨 RECONSTRUYENDO CHROMADB...")
print(f"  Chunks a indexar: {chunks_with_embeddings}")

vector_store = VectorStore(
    collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
    persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
)

chunks_to_index = Chunk.objects.exclude(embedding_vector__isnull=True).exclude(embedding_vector='')

# Process in batches
batch_size = 100
total_indexed = 0

for i in range(0, chunks_to_index.count(), batch_size):
    batch = chunks_to_index[i:i+batch_size]
    
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in batch:
        try:
            import json
            # Check if embedding_vector is already a list or a JSON string
            if isinstance(chunk.embedding_vector, list):
                embedding = chunk.embedding_vector
            elif isinstance(chunk.embedding_vector, str):
                embedding = json.loads(chunk.embedding_vector)
            else:
                print(f"  ⚠️  Tipo inesperado para embedding_vector en chunk {chunk.chunk_id}: {type(chunk.embedding_vector)}")
                continue
            
            ids.append(str(chunk.chunk_id))
            embeddings.append(embedding)
            documents.append(chunk.content)
            metadatas.append({
                'chunk_id': str(chunk.chunk_id),
                'document_id': str(chunk.document.id),
                'document_title': chunk.document.title,
                'page': chunk.metadata.get('page') if chunk.metadata else None
            })
        except Exception as e:
            print(f"  ⚠️  Error con chunk {chunk.chunk_id}: {e}")
            continue
    
    if ids:
        success = vector_store.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        if success:
            # Update Django
            for chunk_id in ids:
                Chunk.objects.filter(chunk_id=chunk_id).update(
                    indexed_in_chromadb=True,
                    chromadb_id=chunk_id
                )
            total_indexed += len(ids)
            print(f"  ✓ Indexados {total_indexed}/{chunks_with_embeddings}")

print(f"\n{'=' * 80}")
print(f"✓ RECONSTRUCCIÓN COMPLETADA")
print(f"  Total indexado: {total_indexed} chunks")
print(f"{'=' * 80}")
