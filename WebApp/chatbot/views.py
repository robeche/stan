from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import uuid
import requests
from .models import Conversation, Message
from admin_panel.models import Chunk
from tools.vector_store import VectorStore
from tools.embeddings import EmbeddingGenerator

# Ollama configuration from settings
OLLAMA_CONFIG = settings.OLLAMA_CONFIG
OLLAMA_URL = OLLAMA_CONFIG['URL']
OLLAMA_MODEL = OLLAMA_CONFIG['MODEL']


def chatbot_interface(request):
    """
    Main chatbot interface view
    """
    # Get or create session_id
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    
    # Get or create conversation
    conversation, created = Conversation.objects.get_or_create(session_id=session_id)
    
    # Get conversation history
    messages = conversation.messages.all()
    
    context = {
        'conversation': conversation,
        'messages': messages,
    }
    
    return render(request, 'chatbot/chat.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    """
    API endpoint to send a message and get a response
    """
    try:
        data = json.loads(request.body)
        user_message = data.get('message', '').strip()
        session_id = request.session.get('chat_session_id')
        
        if not user_message:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        if not session_id:
            session_id = str(uuid.uuid4())
            request.session['chat_session_id'] = session_id
        
        # Get or create conversation
        conversation, _ = Conversation.objects.get_or_create(session_id=session_id)
        
        # Save user message
        user_msg = Message.objects.create(
            conversation=conversation,
            message_type='user',
            content=user_message
        )
        
        # Generate response
        try:
            response_text, retrieved_chunks = generate_response(user_message)
        except Exception as e:
            # If generation fails, still return an error message to the user
            import traceback
            traceback.print_exc()
            response_text = f"Lo siento, ocurrió un error al procesar tu pregunta: {str(e)}"
            retrieved_chunks = []
        
        # Save assistant message
        assistant_msg = Message.objects.create(
            conversation=conversation,
            message_type='assistant',
            content=response_text
        )
        
        # Link retrieved chunks to the message
        if retrieved_chunks:
            assistant_msg.retrieved_chunks.set(retrieved_chunks)
        
        return JsonResponse({
            'success': True,
            'user_message': {
                'id': user_msg.id,
                'content': user_msg.content,
                'created_at': user_msg.created_at.isoformat(),
            },
            'assistant_message': {
                'id': assistant_msg.id,
                'content': assistant_msg.content,
                'created_at': assistant_msg.created_at.isoformat(),
                'sources': [
                    {
                        'chunk_id': chunk.id,
                        'document_name': chunk.document.title,
                        'preview': chunk.content[:200]
                    }
                    for chunk in retrieved_chunks[:5]  # Return top 5 sources
                ]
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e), 'success': False}, status=500)


def generate_response(query):
    """
    Generate a response using RAG pipeline:
    1. Generate query embedding
    2. Retrieve relevant chunks from ChromaDB
    3. Generate response using Ollama LLM with retrieved context
    """
    try:
        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator()
        
        # Generate query embedding
        query_embedding = embedding_gen.generate_single_embedding(query)
        
        # Initialize vector store with Django settings
        vector_store = VectorStore(
            collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
            persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
        )
        
        # Query vector store
        results = vector_store.query(
            query_embedding=query_embedding,
            n_results=5
        )
        
        # Extract chunk IDs from results (ChromaDB returns them in metadata)
        chunk_ids = []
        if results and 'results' in results and len(results['results']) > 0:
            for result in results['results']:
                if 'metadata' in result and 'chunk_id' in result['metadata']:
                    chunk_ids.append(result['metadata']['chunk_id'])
            print(f"DEBUG: Extracted {len(chunk_ids)} chunk IDs: {chunk_ids}")
        
        # Get chunk objects from database
        chunks = []
        if chunk_ids:
            chunks = list(Chunk.objects.filter(chunk_id__in=chunk_ids))
            print(f"DEBUG: Found {len(chunks)} chunks in Django DB")
        
        # Build context from retrieved chunks
        if chunks:
            # Collect tables and images from chunks
            tables_info = []
            images_info = []
            documents_processed = set()
            
            for chunk in chunks[:3]:
                # Get all tables from the document (not filtered by page)
                if chunk.document.id not in documents_processed:
                    documents_processed.add(chunk.document.id)
                    for table in chunk.document.tables.all():
                        # Use descriptive caption from filename if caption is generic
                        caption = table.caption
                        if not caption or caption.startswith('Tabla '):
                            # Extract descriptive name from filename
                            import os
                            filename = os.path.basename(table.table_path)
                            # Remove extension and clean up
                            caption = filename.replace('.png', '').replace('_', ' ').strip()
                        
                        tables_info.append({
                            'id': table.id,
                            'url': table.table_image.url,
                            'caption': caption,
                            'reference': f"TABLA_{table.id}"
                        })
                
                # Images can be page-specific
                if 'page' in chunk.metadata:
                    page_num = chunk.metadata.get('page')
                    for img in chunk.document.images.all():
                        if img.page and img.page.page_number == page_num:
                            # Use descriptive caption from filename if caption is generic
                            caption = img.caption
                            if not caption or caption.startswith('Imagen '):
                                # Extract descriptive name from filename
                                import os
                                filename = os.path.basename(img.image_path)
                                # Remove extension and clean up
                                caption = filename.replace('.png', '').replace('_', ' ').strip()
                            
                            images_info.append({
                                'id': img.id,
                                'url': img.image_file.url,
                                'caption': caption,
                                'reference': f"IMAGEN_{img.id}"
                            })
            
            context = "\n\n".join([
                f"Documento: {chunk.document.title}\n{chunk.content}" 
                for chunk in chunks[:3]  # Use top 3 chunks
            ])
            
            # Add tables/images references to context
            if tables_info:
                context += "\n\nTablas disponibles:\n"
                context += "\n".join([f"- {t['reference']}: {t['caption']}" for t in tables_info])
                print(f"DEBUG: Tables info: {tables_info}")
            if images_info:
                context += "\n\nImágenes disponibles:\n"
                context += "\n".join([f"- {i['reference']}: {i['caption']}" for i in images_info])
                print(f"DEBUG: Images info: {images_info}")
            
            # Build prompt for Ollama
            prompt = f"""You are an expert assistant that answers questions based on technical documents.

**CRITICAL: Respond in the SAME LANGUAGE as the user's question. If the question is in English, respond in English. If in Spanish, respond in Spanish.**

Document context:
{context}

User's question: {query}

Instructions:
- Answer clearly and concisely
- Use ONLY the information provided in the context
- If the information is not in the context, state it clearly
- Cite the document when relevant
- IMPORTANT: If you mention a table or image, include its EXACT reference as listed (example: TABLA_15 or IMAGEN_8). DO NOT use alternative descriptions.

Response:"""
            
            # Call Ollama API
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": OLLAMA_CONFIG.get('TEMPERATURE', 0.7),
                            "top_p": OLLAMA_CONFIG.get('TOP_P', 0.9),
                        }
                    },
                    timeout=OLLAMA_CONFIG.get('TIMEOUT', 60)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                    
                    # Post-process response to inject images/tables
                    # First, try exact reference matches
                    for table in tables_info:
                        marker = table['reference']
                        if marker in answer:
                            img_html = f'<div class="embedded-table"><img src="{table["url"]}" alt="{table["caption"]}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;"><div style="font-size: 0.85rem; color: #666; font-style: italic;">{table["caption"]}</div></div>'
                            answer = answer.replace(marker, img_html)
                    
                    for img in images_info:
                        marker = img['reference']
                        if marker in answer:
                            img_html = f'<div class="embedded-image"><img src="{img["url"]}" alt="{img["caption"]}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;"><div style="font-size: 0.85rem; color: #666; font-style: italic;">{img["caption"]}</div></div>'
                            answer = answer.replace(marker, img_html)
                    
                    # Also try to match natural language patterns like "figura 3-5", "imagen 3-6", "tabla 1-1"
                    import re
                    
                    # Build a map of caption keywords to images/tables
                    for table in tables_info:
                        # Extract patterns like "1-1" from caption
                        patterns = re.findall(r'\d+-\d+', table['caption'])
                        for pattern in patterns:
                            # Look for this pattern in the answer
                            regex = re.compile(rf'\b(tabla|table|figura|figure)\s+{re.escape(pattern)}\b', re.IGNORECASE)
                            if regex.search(answer):
                                img_html = f'<div class="embedded-table"><img src="{table["url"]}" alt="{table["caption"]}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;"><div style="font-size: 0.85rem; color: #666; font-style: italic;">{table["caption"]}</div></div>'
                                # Append at the end to avoid breaking the text
                                if img_html not in answer:
                                    answer += '\n\n' + img_html
                    
                    for img in images_info:
                        # Try to match patterns in caption or just look for generic figure mentions
                        patterns = re.findall(r'\d+-\d+|\bfig\d+', img['caption'], re.IGNORECASE)
                        matched = False
                        
                        for pattern in patterns:
                            regex = re.compile(rf'\b(imagen|image|figura|figure)\s+{re.escape(pattern)}\b', re.IGNORECASE)
                            if regex.search(answer):
                                matched = True
                                break
                        
                        # Also check if answer mentions figures and this image caption contains "fig"
                        if not matched and 'fig' in img['caption'].lower():
                            if re.search(r'\b(figura|figure)\s+\d+[-‑]\d+', answer, re.IGNORECASE):
                                matched = True
                        
                        if matched:
                            img_html = f'<div class="embedded-image"><img src="{img["url"]}" alt="{img["caption"]}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;"><div style="font-size: 0.85rem; color: #666; font-style: italic;">{img["caption"]}</div></div>'
                            if img_html not in answer:
                                answer += '\n\n' + img_html
                    
                    return answer, chunks
                else:
                    return f"Error al comunicarse con Ollama: {response.status_code}", chunks
                    
            except requests.exceptions.ConnectionError:
                return "Error: No se pudo conectar al servidor Ollama. Asegúrate de que esté corriendo en http://localhost:11434", chunks
            except requests.exceptions.Timeout:
                return "Error: Tiempo de espera agotado. El modelo está tardando demasiado en responder.", chunks
            except Exception as e:
                return f"Error al llamar a Ollama: {str(e)}", chunks
        else:
            return "No encontré información relevante en la base de datos para responder a tu consulta.", []
        
    except Exception as e:
        return f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}", []


@require_http_methods(["POST"])
def new_conversation(request):
    """
    Start a new conversation
    """
    # Generate new session ID
    session_id = str(uuid.uuid4())
    request.session['chat_session_id'] = session_id
    
    return JsonResponse({
        'success': True,
        'message': 'New conversation started'
    })
