"""
Script para verificar las rutas de las páginas anotadas
"""
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

from admin_panel.models import Document, Page

doc = Document.objects.last()
if doc:
    print(f"Documento: {doc.title}")
    print(f"Total páginas en DB: {doc.pages.count()}")
    print("\nPrimeras 3 páginas:")
    for page in doc.pages.all()[:3]:
        print(f"\nPágina {page.page_number}:")
        print(f"  annotated_markdown_file: {page.annotated_markdown_file}")
        print(f"  has_annotated_image: {page.has_annotated_image()}")
