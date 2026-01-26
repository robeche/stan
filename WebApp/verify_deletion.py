"""
Verify ChromaDB cleanup after document deletion
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

from django.conf import settings
from admin_panel.models import Document, Chunk
from tools.vector_store import VectorStore

# Check Django database
print("=" * 60)
print("ESTADO DE LA BASE DE DATOS")
print("=" * 60)

total_docs = Document.objects.count()
total_chunks = Chunk.objects.count()
indexed_chunks = Chunk.objects.filter(indexed_in_chromadb=True).count()

print(f"\nDocumentos en Django: {total_docs}")
print(f"Chunks totales: {total_chunks}")
print(f"Chunks marcados como indexados: {indexed_chunks}")

if total_docs > 0:
    print("\nDocumentos existentes:")
    for doc in Document.objects.all():
        doc_chunks = Chunk.objects.filter(document=doc).count()
        doc_indexed = Chunk.objects.filter(document=doc, indexed_in_chromadb=True).count()
        print(f"  - {doc.title}: {doc_chunks} chunks ({doc_indexed} indexados)")

# Check ChromaDB
print("\n" + "=" * 60)
print("ESTADO DE CHROMADB")
print("=" * 60)

try:
    vector_store = VectorStore(
        collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
        persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
    )
    
    # Get collection count from ChromaDB (using the wrapper's internal store)
    chromadb_count = vector_store.store.collection.count()
    
    # Verify consistency
    print("\n" + "=" * 60)
    print("VERIFICACIÓN DE CONSISTENCIA")
    print("=" * 60)
    
    if indexed_chunks == chromadb_count:
        print(f"\n✓ CONSISTENTE: Django ({indexed_chunks}) == ChromaDB ({chromadb_count})")
    else:
        print(f"\n✗ INCONSISTENTE:")
        print(f"  Django dice: {indexed_chunks} chunks indexados")
        print(f"  ChromaDB tiene: {chromadb_count} vectores")
        print(f"  Diferencia: {abs(indexed_chunks - chromadb_count)}")
        
        if indexed_chunks > chromadb_count:
            print("\n  → ChromaDB tiene MENOS vectores de los esperados")
            print("  → Algunos chunks no se eliminaron correctamente de Django")
        else:
            print("\n  → ChromaDB tiene MÁS vectores de los esperados")
            print("  → Algunos vectores no se eliminaron de ChromaDB")
            
        print("\n  Ejecuta: python rebuild_chromadb.py")

except Exception as e:
    print(f"\n✗ Error al consultar ChromaDB: {str(e)}")
    print("   La base de datos puede estar vacía o corrupta")

print("\n" + "=" * 60)
