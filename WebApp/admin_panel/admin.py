from django.contrib import admin
from .models import Document, Page, Image, Table, Chunk, ProcessingLog


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'status', 'progress_percentage', 'uploaded_by', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['title', 'original_filename']
    readonly_fields = ['celery_task_id', 'created_at', 'updated_at', 'processing_started_at', 'processing_completed_at']


@admin.register(Page)
class PageAdmin(admin.ModelAdmin):
    list_display = ['document', 'page_number', 'created_at']
    list_filter = ['created_at']
    search_fields = ['document__title', 'content']


@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['document', 'position_in_document', 'caption', 'created_at']
    list_filter = ['created_at']
    search_fields = ['document__title', 'caption']


@admin.register(Table)
class TableAdmin(admin.ModelAdmin):
    list_display = ['document', 'position_in_document', 'caption', 'created_at']
    list_filter = ['created_at']
    search_fields = ['document__title', 'caption']


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ['document', 'chunk_id', 'chunk_index', 'indexed_in_chromadb', 'created_at']
    list_filter = ['indexed_in_chromadb', 'created_at']
    search_fields = ['document__title', 'chunk_id', 'content']


@admin.register(ProcessingLog)
class ProcessingLogAdmin(admin.ModelAdmin):
    list_display = ['document', 'level', 'stage', 'message', 'created_at']
    list_filter = ['level', 'stage', 'created_at']
    search_fields = ['document__title', 'message']
