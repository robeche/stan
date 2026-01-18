"""
Celery tasks for document processing pipeline.
Integrates: parsing (Nemotron) → chunking → embeddings → ChromaDB
"""
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from celery import shared_task
from django.conf import settings
from django.utils import timezone

from admin_panel.models import Document, Page, Image, Table, Chunk, ProcessingLog

# Import tools from local tools package
from tools.parser import parse_document
from tools.chunker import DocumentChunker
from tools.embeddings import EmbeddingGenerator
from tools.vector_store import VectorStore


def log_processing(document, stage, message, level='info', details=None):
    """Helper function to log processing steps."""
    ProcessingLog.objects.create(
        document=document,
        stage=stage,
        message=message,
        level=level,
        details=details
    )
    print(f"[{stage.upper()}] {message}")


@shared_task(bind=True)
def process_document(self, document_id):
    """
    Main task that processes a document through the entire RAG pipeline.
    
    Pipeline stages:
    1. Parsing with Nemotron (extract pages, images, tables)
    2. Chunking (divide document into fragments)
    3. Embeddings (generate vectors with BGE-M3)
    4. Indexing (store in ChromaDB)
    """
    try:
        document = Document.objects.get(id=document_id)
        document.status = 'parsing'
        document.progress_percentage = 0
        document.processing_started_at = timezone.now()
        document.celery_task_id = self.request.id
        document.save()
        
        log_processing(document, 'init', 'Iniciando procesamiento del documento')
        
        # ============================================================
        # STAGE 1: PARSING with Nemotron
        # ============================================================
        document.status = 'parsing'
        document.progress_percentage = 10
        document.save()
        
        log_processing(document, 'parsing', 'Iniciando parsing con Nemotron...')
        
        try:
            # Prepare output directory
            doc_name = Path(document.original_filename).stem
            output_dir = Path(settings.RAG_CONFIG['PARSING']['OUTPUT_DIR']) / doc_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get file path
            input_file = Path(settings.MEDIA_ROOT) / str(document.file)
            
            # Process document with Nemotron
            log_processing(document, 'parsing', f'Procesando: {input_file}')
            
            result = parse_document(
                input_path=str(input_file),
                output_dir=str(output_dir),
                dpi=300
            )
            
            document.output_directory = result['output_dir']
            document.progress_percentage = 25
            document.save()
            
            # Store pages
            pages_dir = output_dir / 'pages'
            annotated_dir = output_dir / 'annotated_pages'
            raw_output_dir = output_dir / 'raw_output'
            
            log_processing(document, 'parsing', f'Buscando páginas en: {raw_output_dir}', level='info')
            log_processing(document, 'parsing', f'Directorio raw_output existe: {raw_output_dir.exists()}', level='info')
            
            # Read text content from raw_output (page_X_raw.txt)
            page_files = []
            if raw_output_dir.exists():
                page_files = sorted(raw_output_dir.glob('page_*_raw.txt'))
                log_processing(document, 'parsing', f'Archivos .txt encontrados en raw_output/: {len(page_files)}', level='info')
            
            if page_files:
                annotated_count = 0
                for page_file in page_files:
                    # Extract page number from filename
                    try:
                        # Handle both page_1.md and page_1_raw.txt formats
                        filename = page_file.stem
                        if '_raw' in filename:
                            # Extract from page_1_raw -> 1
                            page_num = int(filename.split('_')[1])
                        else:
                            # Extract from page_1 -> 1
                            page_num = int(filename.split('_')[1])
                    except (ValueError, IndexError) as e:
                        log_processing(document, 'parsing', f'No se pudo extraer número de página de {page_file.name}: {e}', level='warning')
                        continue
                    
                    with open(page_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for annotated PNG image (not text file)
                    # Annotated pages are PNG images with bounding boxes drawn
                    annotated_png = annotated_dir / f"page_{page_num}_annotated.png"
                    annotated_exists = annotated_png.exists()
                    
                    # Copy annotated image to media directory for serving
                    annotated_relative_path = None
                    if annotated_exists:
                        annotated_count += 1
                        media_annotated_dir = Path(settings.MEDIA_ROOT) / 'annotated_pages' / doc_name
                        media_annotated_dir.mkdir(parents=True, exist_ok=True)
                        
                        media_annotated_path = media_annotated_dir / annotated_png.name
                        shutil.copy2(annotated_png, media_annotated_path)
                        
                        # Store relative path for serving
                        annotated_relative_path = f'annotated_pages/{doc_name}/{annotated_png.name}'
                    
                    Page.objects.create(
                        document=document,
                        page_number=page_num,
                        content=content,
                        raw_markdown_file=str(page_file),
                        annotated_markdown_file=annotated_relative_path
                    )
                
                document.total_pages = len(page_files)
                log_processing(document, 'parsing', f'Extraídas {len(page_files)} páginas')
                
                if annotated_count > 0:
                    log_processing(document, 'parsing', f'Encontradas {annotated_count} páginas anotadas', level='info')
                else:
                    log_processing(document, 'parsing', f'No se encontraron páginas anotadas en {annotated_dir}', level='warning')
            else:
                log_processing(document, 'parsing', 'No se encontraron páginas extraídas', level='warning')
            
            # Store images
            figures_dir = output_dir / 'figures'
            if figures_dir.exists():
                image_files = sorted(figures_dir.glob('*.png'))
                for idx, img_file in enumerate(image_files):
                    # Copy image to media directory for serving
                    media_images_dir = Path(settings.MEDIA_ROOT) / 'extracted_images' / doc_name
                    media_images_dir.mkdir(parents=True, exist_ok=True)
                    
                    media_img_path = media_images_dir / img_file.name
                    shutil.copy2(img_file, media_img_path)
                    
                    # Store relative path for serving
                    relative_path = f'extracted_images/{doc_name}/{img_file.name}'
                    
                    # Extract page number from filename (e.g., docname_p1_fig1.png)
                    page_num = None
                    import re
                    match = re.search(r'_p(\d+)_', img_file.name)
                    if match:
                        page_num = int(match.group(1))
                        try:
                            page_obj = Page.objects.get(document=document, page_number=page_num)
                        except Page.DoesNotExist:
                            page_obj = None
                    else:
                        page_obj = None
                    
                    Image.objects.create(
                        document=document,
                        page=page_obj,
                        image_file=relative_path,
                        image_path=str(img_file),
                        position_in_document=idx
                    )
                
                document.total_images = len(image_files)
                log_processing(document, 'parsing', f'Extraídas {len(image_files)} imágenes')
            
            # Store tables
            tables_dir = output_dir / 'tables'
            if tables_dir.exists():
                table_files = sorted(tables_dir.glob('*.png'))
                for idx, table_file in enumerate(table_files):
                    # Copy table image to media directory for serving
                    media_tables_dir = Path(settings.MEDIA_ROOT) / 'extracted_tables' / doc_name
                    media_tables_dir.mkdir(parents=True, exist_ok=True)
                    
                    media_table_path = media_tables_dir / table_file.name
                    shutil.copy2(table_file, media_table_path)
                    
                    # Store relative path for serving
                    relative_path = f'extracted_tables/{doc_name}/{table_file.name}'
                    
                    # Extract page number from filename (e.g., docname_p1_tab1.png)
                    page_num = None
                    match = re.search(r'_p(\d+)_', table_file.name)
                    if match:
                        page_num = int(match.group(1))
                        try:
                            page_obj = Page.objects.get(document=document, page_number=page_num)
                        except Page.DoesNotExist:
                            page_obj = None
                    else:
                        page_obj = None
                    
                    Table.objects.create(
                        document=document,
                        page=page_obj,
                        table_image=relative_path,
                        table_path=str(table_file),
                        position_in_document=idx
                    )
                
                document.total_tables = len(table_files)
                log_processing(document, 'parsing', f'Extraídas {len(table_files)} tablas')
            
            # Load concatenated markdown
            concat_file = output_dir / 'documento_concatenado.md'
            if concat_file.exists():
                with open(concat_file, 'r', encoding='utf-8') as f:
                    document.concatenated_markdown = f.read()
            
            document.parsing_completed = True
            document.progress_percentage = 30
            document.save()
            
            log_processing(document, 'parsing', 'Parsing completado exitosamente', level='success')
            
        except Exception as e:
            log_processing(document, 'parsing', f'Error en parsing: {str(e)}', level='error')
            raise
        
        # ============================================================
        # STAGE 2: CHUNKING
        # ============================================================
        document.status = 'chunking'
        document.progress_percentage = 35
        document.save()
        
        log_processing(document, 'chunking', 'Iniciando división en fragmentos...')
        
        try:
            chunker = DocumentChunker(
                strategy=settings.RAG_CONFIG['CHUNKING']['STRATEGY'],
                chunk_size=settings.RAG_CONFIG['CHUNKING']['CHUNK_SIZE'],
                overlap=settings.RAG_CONFIG['CHUNKING']['OVERLAP']
            )
            
            # Chunk the concatenated markdown
            if not document.concatenated_markdown:
                raise ValueError("No hay documento concatenado para fragmentar")
            
            chunks = chunker.chunk_document(document.concatenated_markdown)
            
            # Create chunks directory
            chunks_dir = output_dir / 'chunks_json'
            chunks_dir.mkdir(exist_ok=True)
            
            # Save chunks to database and files
            for idx, chunk_data in enumerate(chunks):
                chunk_id = f"chunk_{idx:04d}"
                chunk_file = chunks_dir / f"{chunk_id}.json"
                
                # Save to file
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                
                # Save to database
                Chunk.objects.create(
                    document=document,
                    chunk_id=chunk_id,
                    content=chunk_data['text'],
                    chunk_index=idx,
                    metadata=chunk_data.get('metadata', {})
                )
            
            document.total_chunks = len(chunks)
            document.chunking_completed = True
            document.progress_percentage = 50
            document.save()
            
            log_processing(document, 'chunking', f'Generados {len(chunks)} fragmentos', level='success')
            
        except Exception as e:
            log_processing(document, 'chunking', f'Error en chunking: {str(e)}', level='error')
            raise
        
        # ============================================================
        # STAGE 3: EMBEDDINGS
        # ============================================================
        document.status = 'embedding'
        document.progress_percentage = 55
        document.save()
        
        log_processing(document, 'embedding', 'Generando embeddings con BGE-M3...')
        
        try:
            generator = EmbeddingGenerator(
                model_name=settings.RAG_CONFIG['EMBEDDINGS']['MODEL_NAME'],
                device=settings.RAG_CONFIG['EMBEDDINGS']['DEVICE']
            )
            
            # Generate embeddings for all chunks
            chunks = Chunk.objects.filter(document=document).order_by('chunk_index')
            
            texts = [chunk.content for chunk in chunks]
            embeddings = generator.generate_embeddings(texts)
            
            # Save embeddings directory
            embeddings_dir = output_dir / 'embeddings'
            embeddings_dir.mkdir(exist_ok=True)
            
            # Update chunks with embeddings
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk.embedding_vector = embedding.tolist()
                chunk.embedding_dimension = len(embedding)
                chunk.save()
                
                # Also save to file (for backup/inspection)
                embedding_file = embeddings_dir / f"{chunk.chunk_id}.json"
                with open(embedding_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'chunk_id': chunk.chunk_id,
                        'embedding': chunk.embedding_vector,
                        'metadata': chunk.metadata
                    }, f)
            
            document.embedding_completed = True
            document.progress_percentage = 75
            document.save()
            
            log_processing(document, 'embedding', 
                          f'Generados {len(embeddings)} embeddings de dimensión {len(embeddings[0])}',
                          level='success')
            
        except Exception as e:
            log_processing(document, 'embedding', f'Error generando embeddings: {str(e)}', level='error')
            raise
        
        # ============================================================
        # STAGE 4: CHROMADB INDEXING
        # ============================================================
        document.status = 'indexing'
        document.progress_percentage = 80
        document.save()
        
        log_processing(document, 'indexing', 'Indexando en ChromaDB...')
        
        try:
            vector_store = VectorStore(
                collection_name=settings.RAG_CONFIG['VECTOR_STORE']['COLLECTION_NAME'],
                persist_directory=settings.RAG_CONFIG['VECTOR_STORE']['PERSIST_DIRECTORY']
            )
            
            # Prepare data for ChromaDB
            chunks = Chunk.objects.filter(document=document).order_by('chunk_index')
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for chunk in chunks:
                # Create unique ID combining document and chunk
                chroma_id = f"doc_{document.id}_chunk_{chunk.chunk_index}"
                
                ids.append(chroma_id)
                embeddings.append(chunk.embedding_vector)
                documents.append(chunk.content)
                
                # Prepare metadata (ChromaDB doesn't accept None values)
                metadata = {
                    'document_id': document.id,
                    'document_title': document.title,
                    'chunk_id': chunk.chunk_id,
                    'chunk_index': chunk.chunk_index,
                }
                
                # Add custom metadata from chunking
                for key, value in chunk.metadata.items():
                    if value is not None:
                        metadata[key] = value
                
                metadatas.append(metadata)
                
                # Update chunk with ChromaDB ID
                chunk.chromadb_id = chroma_id
                chunk.indexed_in_chromadb = True
                chunk.save()
            
            # Add to ChromaDB
            vector_store.add_documents(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            document.indexing_completed = True
            document.progress_percentage = 100
            document.status = 'completed'
            document.processing_completed_at = timezone.now()
            document.save()
            
            processing_time = document.get_processing_time()
            log_processing(document, 'indexing', 
                          f'Indexados {len(ids)} fragmentos en ChromaDB. Tiempo total: {processing_time:.2f}s',
                          level='success')
            
        except Exception as e:
            log_processing(document, 'indexing', f'Error indexando en ChromaDB: {str(e)}', level='error')
            raise
        
        return {
            'status': 'completed',
            'document_id': document.id,
            'total_pages': document.total_pages,
            'total_chunks': document.total_chunks,
            'total_images': document.total_images,
            'total_tables': document.total_tables,
            'processing_time': document.get_processing_time()
        }
        
    except Document.DoesNotExist:
        return {'status': 'error', 'message': f'Document {document_id} not found'}
    
    except Exception as e:
        # Handle any unexpected errors
        try:
            document.status = 'failed'
            document.error_message = str(e)
            document.save()
            log_processing(document, 'error', f'Error fatal: {str(e)}', level='error')
        except:
            pass
        
        raise e
