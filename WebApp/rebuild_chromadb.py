"""
Script para reconstruir ChromaDB desde los chunks existentes en Django.
Útil cuando ChromaDB tiene índices corruptos pero Django tiene los datos.
"""
import sys
from pathlib import Path

# Django setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')

import django
django.setup()

from django.conf import settings
from admin_panel.models import Document, Chunk
from tools.vector_store import VectorStore
import shutil

def rebuild_chromadb():
    """Rebuild ChromaDB from existing Django chunks"""
    print("="*80)
    print("RECONSTRUIR CHROMADB DESDE DJANGO")
    print("="*80)
    print()
    
    # Get all indexed chunks
    chunks = Chunk.objects.filter(
        indexed_in_chromadb=True,
        embedding_vector__isnull=False
    ).select_related('document')
    
    if not chunks.exists():
        print("❌ No hay chunks con embeddings en Django")
        print("   Necesitas procesar documentos primero")
        return
    
    print(f"📦 Encontrados {chunks.count()} chunks con embeddings")
    
    # Group by document
    docs_dict = {}
    for chunk in chunks:
        if chunk.document.id not in docs_dict:
            docs_dict[chunk.document.id] = {
                'title': chunk.document.title,
                'chunks': []
            }
        docs_dict[chunk.document.id]['chunks'].append(chunk)
    
    print(f"📄 De {len(docs_dict)} documentos:")
    for doc_id, doc_data in docs_dict.items():
        print(f"   - {doc_data['title']}: {len(doc_data['chunks'])} chunks")
    
    print()
    response = input("¿Reconstruir ChromaDB con estos datos? [SI/no]: ")
    
    if response.strip().upper() not in ['SI', 'S', 'YES', 'Y', '']:
        print("❌ Operación cancelada")
        return
    
    # Delete ChromaDB directory
    chroma_dir = Path(settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY'])
    
    if chroma_dir.exists():
        print(f"\n🗑️  Eliminando ChromaDB corrupto: {chroma_dir}")
        try:
            shutil.rmtree(chroma_dir)
            print("✓ Eliminado")
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    # Recreate ChromaDB
    print(f"\n🔄 Recreando ChromaDB...")
    
    try:
        vector_store = VectorStore(
            collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
            persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
        )
        print("✓ ChromaDB inicializado")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Re-index all chunks
    print(f"\n📊 Indexando {chunks.count()} chunks...")
    
    batch_size = 100
    total = chunks.count()
    
    for i in range(0, total, batch_size):
        batch = list(chunks[i:i+batch_size])
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in batch:
            # Create ChromaDB ID
            chroma_id = f"doc_{chunk.document.id}_chunk_{chunk.chunk_index}"
            
            ids.append(chroma_id)
            embeddings.append(chunk.embedding_vector)
            documents.append(chunk.content)
            
            # Prepare metadata
            metadata = {
                'document_id': chunk.document.id,
                'document_title': chunk.document.title,
                'chunk_id': chunk.chunk_id,
                'chunk_index': chunk.chunk_index,
            }
            
            # Add chunk metadata
            for key, value in chunk.metadata.items():
                if value is not None:
                    metadata[key] = value
            
            metadatas.append(metadata)
            
            # Update chunk with ChromaDB ID
            chunk.chromadb_id = chroma_id
            chunk.save(update_fields=['chromadb_id'])
        
        # Add batch to ChromaDB
        try:
            vector_store.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            progress = min(i + batch_size, total)
            print(f"  Progreso: {progress}/{total} chunks ({int(progress/total*100)}%)")
            
        except Exception as e:
            print(f"❌ Error en batch {i}-{i+batch_size}: {e}")
            return
    
    print()
    print("="*80)
    print("✅ CHROMADB RECONSTRUIDO EXITOSAMENTE")
    print("="*80)
    print(f"✓ {total} chunks indexados")
    print(f"✓ {len(docs_dict)} documentos")
    print()
    print("Ahora puedes usar el chatbot:")
    print("http://localhost:8000/chatbot/")
    print()

if __name__ == "__main__":
    rebuild_chromadb()
