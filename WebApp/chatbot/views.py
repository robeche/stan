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
            response_text = f"Sorry, an error occurred while processing your question: {str(e)}"
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
                        'chunk_id': chunk.chunk_id,
                        'chunk_db_id': chunk.id,
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
        
        # Query expansion: add synonyms for better retrieval
        query_expanded = query
        # Common technical synonyms
        synonyms = {
            'diameter': 'diameter length size',
            'length': 'length diameter size',
            'height': 'height length',
            'width': 'width diameter',
        }
        for term, expansion in synonyms.items():
            if term.lower() in query.lower() and term not in expansion.split()[0]:
                query_expanded = f"{query} {expansion}"
                break
        
        # Generate query embedding (use expanded query for better retrieval)
        query_embedding = embedding_gen.generate_single_embedding(query_expanded)
        
        # Initialize vector store with Django settings
        vector_store = VectorStore(
            collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
            persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
        )
        
        # Query vector store - retrieve more chunks for better coverage
        try:
            results = vector_store.query(
                query_embedding=query_embedding,
                n_results=10  # Increased from 5 to capture more potential matches
            )
        except Exception as e:
            # Handle ChromaDB errors (empty collection, corrupted index, etc.)
            print(f"ERROR: ChromaDB query failed: {str(e)}")
            print(f"This usually means the database is empty or needs re-indexing")
            results = None
        
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
            chunks_raw = list(Chunk.objects.filter(chunk_id__in=chunk_ids))
            print(f"DEBUG: Found {len(chunks_raw)} chunks in Django DB")
            
            # Filter out very small chunks (less than 100 characters)
            chunks_before_filter = len(chunks_raw)
            chunks = [chunk for chunk in chunks_raw if len(chunk.content) >= 100]
            chunks_filtered = chunks_before_filter - len(chunks)
            
            if chunks_filtered > 0:
                print(f"DEBUG: Filtered out {chunks_filtered} chunks with less than 100 characters")
            
            print(f"DEBUG: Query: '{query}' -> Expanded: '{query_expanded}'")
            for i, chunk in enumerate(chunks[:5], 1):
                print(f"DEBUG: Chunk {i} ({chunk.chunk_id}, {len(chunk.content)} chars): {chunk.content[:100]}...")
        
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
                            'reference': f"TABLA_{table.id}",
                            'filename': os.path.basename(table.table_path)
                        })
                        
                        print(f"DEBUG: Table found - ID: {table.id}, Caption: '{caption}', Filename: '{os.path.basename(table.table_path)}')")
                
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
            
            # Use top 2 chunks for more focused context
            context = "\n\n".join([
                f"Documento: {chunk.document.title}\n{chunk.content}" 
                for chunk in chunks[:10]
            ])
            
            # Add tables/images references to context
            available_refs = []
            if tables_info:
                context += "\n\nAvailable tables:\n"
                context += "\n".join([f"- {t['reference']}: {t['caption']}" for t in tables_info])
                available_refs.extend([t['reference'] for t in tables_info])
                print(f"DEBUG: Tables available: {[t['reference'] for t in tables_info]}")
            if images_info:
                context += "\n\nAvailable images:\n"
                context += "\n".join([f"- {i['reference']}: {i['caption']}" for i in images_info])
                available_refs.extend([i['reference'] for i in images_info])
                print(f"DEBUG: Images available: {[i['reference'] for i in images_info]}")
            
            # Build prompt for Ollama
            refs_instruction = ""
            if available_refs:
                refs_instruction = f"\n7. When referencing tables or images, use ONLY these exact references: {', '.join(available_refs)}. DO NOT invent other reference numbers."
            
            prompt = f"""You are a technical documentation assistant. Answer the user's question based on the provided context.

Document context:
{context}

User's question: {query}

INSTRUCTIONS:
1. Answer using the information from the context above
2. The context may contain tables in LaTeX format (\\begin{{tabular}}) - read them carefully
3. If the answer is in a list or table, extract and present it clearly
4. If the exact information is not in the context, say: "I don't have that specific information in the available documents."
5. Reference the document name when relevant
6. For tables/images, ONLY use the references listed in "Available tables" or "Available images" above{refs_instruction}

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
                            "temperature": 0.3,  # Lower temperature for more precise, grounded answers
                            "top_p": 0.9,
                            "num_ctx": 8192,  # Increase context window
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
                        # Extract patterns like "1-1" from caption AND filename
                        patterns = re.findall(r'\d+[-‑]\d+', table['caption'])
                        filename_patterns = re.findall(r'\d+[-‑]\d+', table['filename'])
                        all_patterns = set(patterns + filename_patterns)
                        matched = False
                        
                        for pattern in all_patterns:
                            # Normalize hyphens (both regular - and en-dash ‑)
                            pattern_normalized = pattern.replace('‑', '-')
                            # Look for this pattern in the answer with flexible hyphen matching
                            regex = re.compile(rf'\b(tabla|table)\s+{re.escape(pattern_normalized).replace("-", "[-‑]")}\b', re.IGNORECASE)
                            if regex.search(answer):
                                matched = True
                                print(f"DEBUG: MATCHED table by pattern '{pattern}' in answer")
                                break
                        
                        # Also search in the answer for general table references if not matched yet
                        if not matched:
                            # Look for patterns like "Table 1-1" or "tabla 1‑1" directly in the answer
                            table_refs = re.findall(r'\b(?:tabla|table)\s+(\d+[-‑]\d+)\b', answer, re.IGNORECASE)
                            for ref in table_refs:
                                # Check if this table contains this reference in caption or filename
                                ref_normalized = ref.replace('‑', '-')
                                caption_normalized = table['caption'].replace('‑', '-')
                                filename_normalized = table['filename'].replace('‑', '-')
                                
                                if ref_normalized in caption_normalized or ref_normalized in filename_normalized:
                                    matched = True
                                    print(f"DEBUG: MATCHED table by reference '{ref}' in caption/filename")
                                    break
                        
                        if matched:
                            img_html = f'<div class="embedded-table"><img src="{table["url"]}" alt="{table["caption"]}" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;"><div style="font-size: 0.85rem; color: #666; font-style: italic;">{table["caption"]}</div></div>'
                            # Append at the end to avoid breaking the text
                            if img_html not in answer:
                                answer += '\n\n' + img_html
                                print(f"DEBUG: INSERTED table '{table['caption']}' into answer")
                    
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
                    return f"Error communicating with Ollama: {response.status_code}", chunks
                    
            except requests.exceptions.ConnectionError:
                return "Error: Could not connect to Ollama server. Make sure it's running at http://localhost:11434", chunks
            except requests.exceptions.Timeout:
                return "Error: Request timed out. The model is taking too long to respond.", chunks
            except Exception as e:
                return f"Error calling Ollama: {str(e)}", chunks
        else:
            # No chunks found - still ask the LLM but without context
            print(f"WARNING: No chunks found for query: '{query}'")
            
            prompt = f"""You are an expert assistant for technical documentation.

User's question: {query}

Unfortunately, I could not find specific information in the available documents to answer this question.

Please respond in one of these ways:
1. If this is a general question you can answer based on your knowledge, provide a helpful general answer and clearly state it's not from the specific documents.
2. If it requires specific technical details from documents, explain that you don't have access to this information in the current database.
3. Suggest what type of documentation or information would be needed to answer this question.

Response:"""
            
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
                    return answer, []
                else:
                    return f"No encontré información relevante en la base de datos y hubo un error al generar una respuesta: {response.status_code}", []
                    
            except Exception as e:
                return f"No encontré información relevante en la base de datos. Error al consultar el modelo: {str(e)}", []
        
    except Exception as e:
        return f"Sorry, an error occurred while processing your query: {str(e)}", []


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


@require_http_methods(["GET"])
def get_chunk_details(request, chunk_id):
    """
    API endpoint to get chunk details for modal display
    """
    try:
        chunk = Chunk.objects.get(id=chunk_id)
        
        return JsonResponse({
            'success': True,
            'chunk': {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'document_title': chunk.document.title,
                'embedding_dimension': chunk.embedding_dimension,
                'indexed_in_chromadb': chunk.indexed_in_chromadb,
                'metadata': chunk.metadata,
            }
        })
    except Chunk.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Chunk not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
