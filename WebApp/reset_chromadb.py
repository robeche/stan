"""
Script para resetear completamente la base de datos ChromaDB.
ADVERTENCIA: Esto eliminará todos los chunks indexados.
"""
import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Django setup
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')

import django
django.setup()

from django.conf import settings
from tools.vector_store import VectorStore
from admin_panel.models import Chunk

def reset_chromadb():
    """Reset ChromaDB completely"""
    print("="*80)
    print("RESETEO COMPLETO DE CHROMADB")
    print("="*80)
    print()
    
    # Step 1: Delete ChromaDB directory
    chroma_dir = Path(settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY'])
    
    if chroma_dir.exists():
        print(f"📁 Directorio ChromaDB encontrado: {chroma_dir}")
        print(f"🗑️  Eliminando directorio completo...")
        
        try:
            shutil.rmtree(chroma_dir)
            print(f"✓ Directorio eliminado exitosamente")
        except Exception as e:
            print(f"❌ Error eliminando directorio: {e}")
            return False
    else:
        print(f"ℹ️  El directorio ChromaDB no existe: {chroma_dir}")
    
    print()
    
    # Step 2: Update Django database - mark all chunks as not indexed
    print("📊 Actualizando base de datos Django...")
    
    indexed_chunks = Chunk.objects.filter(indexed_in_chromadb=True)
    count = indexed_chunks.count()
    
    if count > 0:
        indexed_chunks.update(
            indexed_in_chromadb=False,
            chromadb_id=None
        )
        print(f"✓ {count} chunks marcados como no indexados")
    else:
        print(f"ℹ️  No hay chunks marcados como indexados")
    
    print()
    
    # Step 3: Recreate empty ChromaDB
    print("🔄 Recreando ChromaDB vacío...")
    
    try:
        vector_store = VectorStore(
            collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
            persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
        )
        print(f"✓ ChromaDB recreado exitosamente")
        print(f"✓ Colección: {settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME']}")
    except Exception as e:
        print(f"❌ Error recreando ChromaDB: {e}")
        return False
    
    print()
    print("="*80)
    print("✅ RESETEO COMPLETADO EXITOSAMENTE")
    print("="*80)
    print()
    print("📝 Próximos pasos:")
    print("   1. Los documentos en Django siguen existiendo")
    print("   2. Para re-indexarlos, ejecuta el procesamiento de nuevo")
    print("   3. O elimina y vuelve a subir los documentos")
    print()
    
    return True

if __name__ == "__main__":
    import sys
    
    print()
    response = input("⚠️  ¿Estás seguro de que quieres resetear ChromaDB? (escribe 'SI' para confirmar): ")
    
    if response.strip().upper() == 'SI':
        reset_chromadb()
    else:
        print("❌ Operación cancelada")
        sys.exit(0)
