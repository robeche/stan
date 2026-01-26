"""
Script para verificar el estado de ChromaDB y los documentos indexados.
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

def check_chromadb_status():
    """Check ChromaDB status and indexed documents"""
    print("="*80)
    print("VERIFICACIÓN DE ESTADO - CHROMADB Y DOCUMENTOS")
    print("="*80)
    print()
    
    # Check documents in Django
    print("📊 DOCUMENTOS EN DJANGO:")
    print("-" * 80)
    docs = Document.objects.all().order_by('-created_at')
    
    if not docs.exists():
        print("❌ No hay documentos en la base de datos")
        print("   → Sube un documento desde el admin")
        return
    
    for doc in docs:
        print(f"\n📄 {doc.title}")
        print(f"   Estado: {doc.status}")
        print(f"   Progreso: {doc.progress_percentage}%")
        print(f"   Chunks totales: {doc.total_chunks}")
        print(f"   Parsing: {'✓' if doc.parsing_completed else '✗'}")
        print(f"   Chunking: {'✓' if doc.chunking_completed else '✗'}")
        print(f"   Embeddings: {'✓' if doc.embedding_completed else '✗'}")
        print(f"   Indexing: {'✓' if doc.indexing_completed else '✗'}")
        
        if doc.error_message:
            print(f"   ⚠️  Error: {doc.error_message}")
    
    print()
    print("-" * 80)
    
    # Check chunks
    print("\n📦 CHUNKS EN DJANGO:")
    print("-" * 80)
    total_chunks = Chunk.objects.count()
    indexed_chunks = Chunk.objects.filter(indexed_in_chromadb=True).count()
    
    print(f"Total chunks: {total_chunks}")
    print(f"Chunks indexados en ChromaDB: {indexed_chunks}")
    
    if total_chunks > 0 and indexed_chunks == 0:
        print("⚠️  Tienes chunks pero ninguno está marcado como indexado!")
        print("   → Posible problema en la etapa de indexing")
    
    print()
    print("-" * 80)
    
    # Check ChromaDB
    print("\n🗄️  CHROMADB:")
    print("-" * 80)
    
    try:
        vector_store = VectorStore(
            collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
            persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
        )
        
        # Try to count items in collection
        try:
            collection = vector_store.collection
            count = collection.count()
            print(f"✓ ChromaDB conectado")
            print(f"  Colección: {settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME']}")
            print(f"  Items en colección: {count}")
            
            if count == 0:
                print("\n⚠️  ChromaDB está VACÍO")
                print("   Posibles causas:")
                print("   1. El documento aún no terminó de procesarse")
                print("   2. Hubo un error en la etapa de indexing")
                print("   3. Celery no está corriendo")
                
        except Exception as e:
            print(f"⚠️  Error accediendo a la colección: {str(e)}")
            print(f"   La colección puede estar vacía o corrupta")
            
    except Exception as e:
        print(f"❌ Error conectando a ChromaDB: {str(e)}")
    
    print()
    print("="*80)
    print("RECOMENDACIONES:")
    print("="*80)
    
    # Provide recommendations
    docs_pending = Document.objects.exclude(status='completed')
    
    if docs_pending.exists():
        print("\n📋 Tienes documentos pendientes de procesar:")
        for doc in docs_pending:
            print(f"   - {doc.title}: {doc.status} ({doc.progress_percentage}%)")
        print("\n   → Verifica que Celery esté corriendo:")
        print("     cd WebApp")
        print("     start_celery.bat")
    
    if indexed_chunks == 0 and total_chunks > 0:
        print("\n📋 Tienes chunks sin indexar:")
        print("   → Ejecuta: python reindex_documents.py")
    
    if docs.filter(status='completed').exists() and indexed_chunks > 0:
        print("\n✅ Todo parece estar correcto")
        print("   Puedes probar el chatbot en:")
        print("   http://localhost:8000/chatbot/")
    
    print()

if __name__ == "__main__":
    check_chromadb_status()
