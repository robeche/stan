"""
Quick verification script for chatbot setup
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("CHATBOT SETUP VERIFICATION")
print("=" * 80)

# Check Django setup
try:
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
    django.setup()
    print("\n✓ Django setup: OK")
except Exception as e:
    print(f"\n✗ Django setup: FAILED - {e}")
    sys.exit(1)

# Check apps
try:
    from django.apps import apps
    admin_panel = apps.get_app_config('admin_panel')
    chatbot = apps.get_app_config('chatbot')
    print(f"✓ Apps registered: admin_panel, chatbot")
except Exception as e:
    print(f"✗ Apps registration: FAILED - {e}")
    sys.exit(1)

# Check models
try:
    from admin_panel.models import Document, Chunk
    from chatbot.models import Conversation, Message
    print(f"✓ Models imported: Document, Chunk, Conversation, Message")
except Exception as e:
    print(f"✗ Models import: FAILED - {e}")
    sys.exit(1)

# Check database
try:
    doc_count = Document.objects.count()
    chunk_count = Chunk.objects.count()
    conv_count = Conversation.objects.count()
    msg_count = Message.objects.count()
    print(f"✓ Database connected:")
    print(f"  - Documents: {doc_count}")
    print(f"  - Chunks: {chunk_count}")
    print(f"  - Conversations: {conv_count}")
    print(f"  - Messages: {msg_count}")
except Exception as e:
    print(f"✗ Database access: FAILED - {e}")
    sys.exit(1)

# Check tools
try:
    from tools.embeddings import EmbeddingGenerator
    from tools.vector_store import VectorStore
    print(f"✓ RAG tools imported: EmbeddingGenerator, VectorStore")
except Exception as e:
    print(f"✗ Tools import: FAILED - {e}")
    sys.exit(1)

# Check ChromaDB
try:
    import chromadb
    chroma_path = os.path.join(os.path.dirname(__file__), 'chroma_db')
    client = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()
    print(f"✓ ChromaDB connected: {len(collections)} collections")
    if collections:
        for col in collections:
            print(f"  - {col.name}: {col.count()} vectors")
except Exception as e:
    print(f"✗ ChromaDB access: FAILED - {e}")

# Check URLs
try:
    from django.urls import get_resolver
    resolver = get_resolver()
    urls = [
        ('chatbot:chat', 'Chatbot interface'),
        ('chatbot:send_message', 'Send message API'),
        ('chatbot:new_conversation', 'New conversation API'),
        ('admin_panel:dashboard', 'Admin dashboard'),
        ('admin_panel:document_list', 'Document list'),
    ]
    print(f"\n✓ URL patterns registered:")
    for url_name, description in urls:
        try:
            resolver.reverse(url_name)
            print(f"  - {url_name}: {description}")
        except:
            print(f"  ✗ {url_name}: NOT FOUND")
except Exception as e:
    print(f"✗ URL check: FAILED - {e}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)

# Summary
print("\n📊 SUMMARY:")
if chunk_count > 0:
    print("✓ System is ready for queries")
    print(f"✓ {chunk_count} chunks available for search")
else:
    print("⚠ No chunks found - please process documents in admin panel first")

print("\n🚀 TO START THE CHATBOT:")
print("1. Make sure Redis is running: redis-server")
print("2. Start Celery worker: celery -A rag_project worker -l info --pool=solo")
print("3. Start Django server: python manage.py runserver")
print("4. Visit: http://localhost:8000/")

print("\n📝 TO TEST:")
print("python test_chatbot.py")
