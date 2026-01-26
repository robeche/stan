"""
Script para re-indexar documentos existentes en ChromaDB.
Útil cuando has reseteado ChromaDB pero los documentos siguen en Django.
"""
import sys
from pathlib import Path

# Django setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')

import django
django.setup()

from admin_panel.models import Document
from admin_panel.tasks import process_document

def reindex_all_documents():
    """Re-index all completed documents"""
    print("="*80)
    print("RE-INDEXACIÓN DE DOCUMENTOS EN CHROMADB")
    print("="*80)
    print()
    
    # Get all completed documents
    completed_docs = Document.objects.filter(status='completed')
    
    if not completed_docs.exists():
        print("ℹ️  No hay documentos completados para re-indexar")
        print()
        print("Opciones:")
        print("  1. Sube nuevos documentos desde el admin")
        print("  2. Verifica que tienes documentos con status='completed'")
        return
    
    print(f"📊 Encontrados {completed_docs.count()} documentos completados")
    print()
    
    for doc in completed_docs:
        print(f"📄 {doc.title}")
        print(f"   - Chunks: {doc.total_chunks}")
        print(f"   - Páginas: {doc.total_pages}")
    
    print()
    response = input("¿Re-procesar estos documentos? (esto puede tardar) [SI/no]: ")
    
    if response.strip().upper() not in ['SI', 'S', 'YES', 'Y', '']:
        print("❌ Operación cancelada")
        return
    
    print()
    print("🔄 Iniciando re-procesamiento...")
    print()
    
    for doc in completed_docs:
        print(f"Procesando: {doc.title}...")
        try:
            # Reset status to trigger full reprocessing
            doc.status = 'uploaded'
            doc.indexing_completed = False
            doc.save()
            
            # Trigger processing task
            result = process_document.delay(doc.id)
            print(f"  ✓ Tarea iniciada: {result.id}")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    print()
    print("="*80)
    print("✅ TAREAS INICIADAS")
    print("="*80)
    print()
    print("Monitorea el progreso en:")
    print("  - Admin Django: http://localhost:8000/admin/admin_panel/document/")
    print("  - Consola de Celery")
    print()

if __name__ == "__main__":
    reindex_all_documents()
