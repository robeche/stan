"""
Test script for chatbot functionality with Ollama integration
Run from WebApp directory: python test_chatbot.py
"""

import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
django.setup()

from chatbot.views import generate_response
from admin_panel.models import Chunk, Document
from django.conf import settings


def test_generate_response():
    """Test the generate_response function with Ollama"""
    print("=" * 80)
    print("Testing Chatbot Response Generation with Ollama")
    print("=" * 80)
    
    # Check configuration
    ollama_config = settings.OLLAMA_CONFIG
    print(f"\n🔧 Configuración Ollama:")
    print(f"   URL: {ollama_config['URL']}")
    print(f"   Modelo: {ollama_config['MODEL']}")
    print(f"   Temperature: {ollama_config['TEMPERATURE']}")
    
    # Check if we have documents and chunks
    doc_count = Document.objects.count()
    chunk_count = Chunk.objects.count()
    
    print(f"\n📚 Base de datos:")
    print(f"   Documentos: {doc_count}")
    print(f"   Chunks: {chunk_count}")
    
    if chunk_count == 0:
        print("\n❌ No hay chunks. Por favor procesa un documento primero.")
        return
    
    # Test queries
    test_queries = [
        "¿Cuáles son las propiedades principales de la turbina NREL?",
        "¿Qué dimensiones tiene el rotor?",
        "Resume la información sobre las palas",
    ]
    
    print("\n" + "=" * 80)
    print("Probando Consultas con IA")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Pregunta {i}] {query}")
        print("-" * 80)
        
        try:
            print("⏳ Procesando (puede tardar 10-30 segundos)...")
            response, chunks = generate_response(query)
            
            print(f"\n🤖 Respuesta del LLM:")
            print(response)
            
            print(f"\n📚 Chunks usados: {len(chunks)}")
            if chunks:
                print("\n🔍 Fuentes:")
                for j, chunk in enumerate(chunks[:3], 1):
                    print(f"   {j}. {chunk.document.name} (Chunk {chunk.chunk_index})")
        
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        if i < len(test_queries):
            print("\n" + "-" * 80)
    
    print("\n" + "=" * 80)
    print("✓ Test completado!")
    print("=" * 80)
    print("\n💡 Ahora puedes usar el chatbot en: http://localhost:8000/")


if __name__ == "__main__":
    test_generate_response()
