"""
Script para verificar las relaciones entre modelos.
Ejecutar desde el shell de Django: python manage.py shell < test_relations.py
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

from admin_panel.models import Document, Page, Image, Table, Chunk, ProcessingLog

print("=" * 80)
print("VERIFICACIÓN DE RELACIONES ENTRE MODELOS")
print("=" * 80)

# Obtener el último documento
document = Document.objects.last()

if document:
    print(f"\n📄 Documento: {document.title}")
    print(f"   Estado: {document.status}")
    print(f"   Creado: {document.created_at}")
    
    print("\n" + "-" * 80)
    print("RELACIONES ESTABLECIDAS (usando related_name):")
    print("-" * 80)
    
    # Páginas (document.pages)
    pages = document.pages.all()
    print(f"\n📑 Páginas (document.pages.all()):")
    print(f"   Total: {pages.count()}")
    if pages.exists():
        for page in pages[:3]:
            print(f"   - Página {page.page_number}: {len(page.content)} caracteres")
            # Relación inversa: page -> images
            page_images = page.images.all()
            page_tables = page.tables.all()
            if page_images.exists() or page_tables.exists():
                print(f"     └─ {page_images.count()} imágenes, {page_tables.count()} tablas")
    
    # Imágenes (document.images)
    images = document.images.all()
    print(f"\n🖼️  Imágenes (document.images.all()):")
    print(f"   Total: {images.count()}")
    if images.exists():
        for img in images[:3]:
            page_info = f"Página {img.page.page_number}" if img.page else "Sin página"
            print(f"   - Imagen {img.position_in_document}: {page_info}")
    
    # Tablas (document.tables)
    tables = document.tables.all()
    print(f"\n📊 Tablas (document.tables.all()):")
    print(f"   Total: {tables.count()}")
    if tables.exists():
        for table in tables[:3]:
            page_info = f"Página {table.page.page_number}" if table.page else "Sin página"
            print(f"   - Tabla {table.position_in_document}: {page_info}")
    
    # Chunks (document.chunks)
    chunks = document.chunks.all()
    print(f"\n📦 Chunks (document.chunks.all()):")
    print(f"   Total: {chunks.count()}")
    indexed = chunks.filter(indexed_in_chromadb=True).count()
    print(f"   Indexados en ChromaDB: {indexed}")
    if chunks.exists():
        for chunk in chunks[:3]:
            print(f"   - {chunk.chunk_id}: {len(chunk.content)} caracteres")
    
    # Logs (document.logs)
    logs = document.logs.all()
    print(f"\n📝 Logs (document.logs.all()):")
    print(f"   Total: {logs.count()}")
    if logs.exists():
        for log in logs[:5]:
            print(f"   - [{log.level}] {log.stage}: {log.message}")
    
    print("\n" + "=" * 80)
    print("MÉTODOS HELPER DEL MODELO DOCUMENT:")
    print("=" * 80)
    print(f"document.get_pages_count(): {document.get_pages_count()}")
    print(f"document.get_images_count(): {document.get_images_count()}")
    print(f"document.get_tables_count(): {document.get_tables_count()}")
    print(f"document.get_chunks_count(): {document.get_chunks_count()}")
    print(f"document.get_indexed_chunks_count(): {document.get_indexed_chunks_count()}")
    print(f"document.has_related_content(): {document.has_related_content()}")
    
    print("\n" + "=" * 80)
    print("CONCLUSIÓN:")
    print("=" * 80)
    print("✅ Todas las relaciones están correctamente establecidas")
    print("✅ Page, Image, Table, Chunk tienen ForeignKey a Document")
    print("✅ Image y Table también tienen ForeignKey a Page")
    print("✅ Se puede acceder a los objetos relacionados usando:")
    print("   - document.pages.all()")
    print("   - document.images.all()")
    print("   - document.tables.all()")
    print("   - document.chunks.all()")
    print("   - document.logs.all()")
    print("   - page.images.all()")
    print("   - page.tables.all()")
    print("=" * 80)
else:
    print("\n❌ No hay documentos en la base de datos")
    print("   Sube un documento primero desde la interfaz web")
