"""
Django signals for automatic ChromaDB cleanup when documents are deleted.
"""
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from django.conf import settings
from pathlib import Path
import shutil

from .models import Document, Chunk
from tools.vector_store import VectorStore


@receiver(pre_delete, sender=Document)
def delete_document_from_chromadb(sender, instance, **kwargs):
    """
    Delete all chunks from ChromaDB when a Document is deleted.
    Also cleans up associated files.
    
    Uses pre_delete instead of post_delete because Django deletes
    related chunks (via ForeignKey cascade) before post_delete fires.
    """
    print(f"\n{'='*60}")
    print(f"SEÑAL pre_delete activada para: {instance.title}")
    print(f"{'='*60}")
    
    try:
        # Delete chunks from ChromaDB BEFORE Django deletes them
        chunks = Chunk.objects.filter(document=instance, indexed_in_chromadb=True)
        chunks_count = chunks.count()
        
        print(f"📊 Chunks encontrados en Django: {chunks_count}")
        
        if chunks.exists():
            vector_store = VectorStore(
                collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
                persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
            )
            
            # Get all ChromaDB IDs
            chromadb_ids = [chunk.chromadb_id for chunk in chunks if chunk.chromadb_id]
            
            print(f"🔑 ChromaDB IDs a eliminar: {len(chromadb_ids)}")
            
            if chromadb_ids:
                # Delete from ChromaDB
                success = vector_store.delete_by_ids(ids=chromadb_ids)
                if success:
                    print(f"✓ Eliminados {len(chromadb_ids)} chunks de ChromaDB para documento: {instance.title}")
                else:
                    print(f"❌ Error al eliminar chunks de ChromaDB")
            else:
                print(f"⚠️  No se encontraron chromadb_ids válidos")
        else:
            print(f"ℹ️  No hay chunks indexados para eliminar")
        
        # Clean up output directory if it exists
        if instance.output_directory:
            output_path = Path(instance.output_directory)
            if output_path.exists():
                shutil.rmtree(output_path)
                print(f"✓ Eliminado directorio de salida: {output_path}")
        
        # Clean up uploaded file
        if instance.file:
            file_path = Path(settings.MEDIA_ROOT) / str(instance.file)
            if file_path.exists():
                file_path.unlink()
                print(f"✓ Eliminado archivo original: {file_path.name}")
        
    except Exception as e:
        print(f"⚠️  Error eliminando chunks de ChromaDB: {str(e)}")
        # Don't raise exception - allow document deletion to proceed
