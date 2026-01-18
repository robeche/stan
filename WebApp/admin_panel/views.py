"""
Views for admin_panel application.
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db.models import Count, Q
from django.conf import settings
from celery.result import AsyncResult
from pathlib import Path

from .models import Document, Page, Image, Table, Chunk, ProcessingLog
from .tasks import process_document
from .forms import DocumentUploadForm


@staff_member_required
def model_diagram(request):
    """Display database model relationships diagram."""
    return render(request, 'admin_panel/model_diagram.html')


@staff_member_required
def dashboard(request):
    """Main dashboard for admin panel."""
    
    # Get statistics
    total_documents = Document.objects.count()
    completed_documents = Document.objects.filter(status='completed').count()
    processing_documents = Document.objects.exclude(
        status__in=['completed', 'failed']
    ).count()
    failed_documents = Document.objects.filter(status='failed').count()
    
    total_chunks = Chunk.objects.count()
    indexed_chunks = Chunk.objects.filter(indexed_in_chromadb=True).count()
    
    # Recent documents
    recent_documents = Document.objects.all()[:10]
    
    # Recent logs
    recent_logs = ProcessingLog.objects.select_related('document').all()[:20]
    
    context = {
        'total_documents': total_documents,
        'completed_documents': completed_documents,
        'processing_documents': processing_documents,
        'failed_documents': failed_documents,
        'total_chunks': total_chunks,
        'indexed_chunks': indexed_chunks,
        'recent_documents': recent_documents,
        'recent_logs': recent_logs,
    }
    
    return render(request, 'admin_panel/dashboard.html', context)


@staff_member_required
def document_list(request):
    """List all documents."""
    
    # Filter by status if provided
    status_filter = request.GET.get('status', None)
    
    documents = Document.objects.all()
    
    if status_filter:
        documents = documents.filter(status=status_filter)
    
    # Add statistics
    documents = documents.annotate(
        pages_count=Count('pages'),
        images_count=Count('images'),
        tables_count=Count('tables'),
        chunks_count=Count('chunks'),
    )
    
    context = {
        'documents': documents,
        'status_filter': status_filter,
    }
    
    return render(request, 'admin_panel/document_list.html', context)


@staff_member_required
@require_http_methods(["GET", "POST"])
def document_upload(request):
    """Upload and process a new document."""
    
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            document = form.save(commit=False)
            document.uploaded_by = request.user
            document.original_filename = request.FILES['file'].name
            
            # Generate title from filename if not provided
            if not document.title:
                document.title = Path(document.original_filename).stem
            
            document.save()
            
            # Start background processing
            task = process_document.delay(document.id)
            
            document.celery_task_id = task.id
            document.save()
            
            return redirect('admin_panel:document_detail', pk=document.id)
    else:
        form = DocumentUploadForm()
    
    context = {
        'form': form,
    }
    
    return render(request, 'admin_panel/document_upload.html', context)


@staff_member_required
def document_detail(request, pk):
    """View document details and processing status."""
    
    document = get_object_or_404(Document, pk=pk)
    
    # Get related objects
    pages = document.pages.all()
    images = document.images.all()[:20]  # Limit to first 20
    tables = document.tables.all()[:20]
    chunks = document.chunks.all()[:50]  # Limit to first 50
    logs = document.logs.all()
    
    context = {
        'document': document,
        'pages': pages,
        'images': images,
        'tables': tables,
        'chunks': chunks,
        'logs': logs,
        'MEDIA_URL': settings.MEDIA_URL,  # Add MEDIA_URL to context
    }
    
    return render(request, 'admin_panel/document_detail.html', context)


@staff_member_required
@require_http_methods(["POST"])
def document_delete(request, pk):
    """Delete a document and all related data."""
    
    document = get_object_or_404(Document, pk=pk)
    
    # Delete associated files
    if document.file:
        try:
            document.file.delete()
        except:
            pass
    
    # Delete the document (cascade will handle related objects)
    document.delete()
    
    return redirect('admin_panel:document_list')


@staff_member_required
@require_http_methods(["POST"])
def document_reprocess(request, pk):
    """Reprocess a failed document."""
    
    document = get_object_or_404(Document, pk=pk)
    
    # Reset document status
    document.status = 'uploaded'
    document.progress_percentage = 0
    document.error_message = None
    document.parsing_completed = False
    document.chunking_completed = False
    document.embedding_completed = False
    document.indexing_completed = False
    document.processing_started_at = None
    document.processing_completed_at = None
    
    # Delete existing related objects to start fresh
    document.pages.all().delete()
    document.images.all().delete()
    document.tables.all().delete()
    document.chunks.all().delete()
    document.logs.all().delete()
    
    document.total_pages = 0
    document.total_chunks = 0
    document.total_images = 0
    document.total_tables = 0
    
    document.save()
    
    # Start background processing again
    task = process_document.delay(document.id)
    
    document.celery_task_id = task.id
    document.save()
    
    return redirect('admin_panel:document_detail', pk=document.id)


@staff_member_required
def task_status_api(request, task_id):
    """API endpoint to check Celery task status."""
    
    task_result = AsyncResult(task_id)
    
    response_data = {
        'task_id': task_id,
        'status': task_result.status,
        'ready': task_result.ready(),
        'successful': task_result.successful() if task_result.ready() else None,
    }
    
    if task_result.ready():
        if task_result.successful():
            response_data['result'] = task_result.result
        else:
            response_data['error'] = str(task_result.info)
    
    return JsonResponse(response_data)


@staff_member_required
def document_status_api(request, document_id):
    """API endpoint to get document processing status."""
    
    try:
        document = Document.objects.get(id=document_id)
        
        # Get recent logs
        recent_logs = document.logs.order_by('-created_at')[:5]
        
        response_data = {
            'document_id': document.id,
            'status': document.status,
            'progress_percentage': document.progress_percentage,
            'error_message': document.error_message,
            'parsing_completed': document.parsing_completed,
            'chunking_completed': document.chunking_completed,
            'embedding_completed': document.embedding_completed,
            'indexing_completed': document.indexing_completed,
            'total_pages': document.total_pages,
            'total_chunks': document.total_chunks,
            'total_images': document.total_images,
            'total_tables': document.total_tables,
            'recent_logs': [
                {
                    'level': log.level,
                    'stage': log.stage,
                    'message': log.message,
                    'created_at': log.created_at.isoformat(),
                }
                for log in recent_logs
            ]
        }
        
        return JsonResponse(response_data)
    
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found'}, status=404)
