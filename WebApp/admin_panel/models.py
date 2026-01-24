"""
Database models for RAG document processing system.
"""
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json


class Document(models.Model):
    """Main document model representing uploaded files."""
    
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('parsing', 'Parsing'),
        ('chunking', 'Chunking'),
        ('embedding', 'Generating embeddings'),
        ('indexing', 'Indexing in ChromaDB'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    title = models.CharField(max_length=500)
    original_filename = models.CharField(max_length=500)
    file = models.FileField(upload_to='documents/%Y/%m/%d/')
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    progress_percentage = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    
    # Processing metadata
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    parsing_completed = models.BooleanField(default=False)
    chunking_completed = models.BooleanField(default=False)
    embedding_completed = models.BooleanField(default=False)
    indexing_completed = models.BooleanField(default=False)
    
    # Statistics
    total_pages = models.IntegerField(default=0)
    total_chunks = models.IntegerField(default=0)
    total_images = models.IntegerField(default=0)
    total_tables = models.IntegerField(default=0)
    
    # Paths to outputs
    output_directory = models.CharField(max_length=1000, blank=True, null=True)
    concatenated_markdown = models.TextField(blank=True, null=True)
    
    # User and timestamp
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_documents')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    processing_started_at = models.DateTimeField(blank=True, null=True)
    processing_completed_at = models.DateTimeField(blank=True, null=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Document'
        verbose_name_plural = 'Documents'
    
    def __str__(self):
        return f"{self.title} ({self.status})"
    
    def get_processing_time(self):
        """Calculate processing time in seconds."""
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return delta.total_seconds()
        return None
    
    def get_pages_count(self):
        """Get actual count of related pages."""
        return self.pages.count()
    
    def get_images_count(self):
        """Get actual count of related images."""
        return self.images.count()
    
    def get_tables_count(self):
        """Get actual count of related tables."""
        return self.tables.count()
    
    def get_chunks_count(self):
        """Get actual count of related chunks."""
        return self.chunks.count()
    
    def get_indexed_chunks_count(self):
        """Get count of chunks indexed in ChromaDB."""
        return self.chunks.filter(indexed_in_chromadb=True).count()
    
    def has_related_content(self):
        """Check if document has any related content."""
        return (self.pages.exists() or 
                self.images.exists() or 
                self.tables.exists() or 
                self.chunks.exists())


class Page(models.Model):
    """Extracted pages from documents."""
    
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='pages')
    page_number = models.IntegerField()
    content = models.TextField()
    
    # Files
    raw_markdown_file = models.CharField(max_length=1000, blank=True, null=True)
    annotated_markdown_file = models.CharField(max_length=1000, blank=True, null=True)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['page_number']
        unique_together = ['document', 'page_number']
        verbose_name = 'Page'
        verbose_name_plural = 'Pages'
    
    def __str__(self):
        return f"{self.document.title} - Página {self.page_number}"
    
    def get_annotated_content(self):
        """Return relative path for annotated PNG image if it exists."""
        return self.annotated_markdown_file
    
    def has_annotated_image(self):
        """Check if annotated PNG image path exists."""
        return bool(self.annotated_markdown_file)


class Image(models.Model):
    """Images extracted from documents."""
    
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='images')
    page = models.ForeignKey(Page, on_delete=models.CASCADE, related_name='images', null=True, blank=True)
    
    image_file = models.ImageField(upload_to='extracted_images/%Y/%m/%d/')
    image_path = models.CharField(max_length=1000)
    
    # Metadata
    caption = models.TextField(blank=True, null=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    position_in_document = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['position_in_document']
        verbose_name = 'Image'
        verbose_name_plural = 'Images'
    
    def __str__(self):
        return f"Imagen {self.position_in_document} - {self.document.title}"


class Table(models.Model):
    """Tables extracted as images from documents."""
    
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='tables')
    page = models.ForeignKey(Page, on_delete=models.CASCADE, related_name='tables', null=True, blank=True)
    
    table_image = models.ImageField(upload_to='extracted_tables/%Y/%m/%d/')
    table_path = models.CharField(max_length=1000)
    
    # Metadata
    caption = models.TextField(blank=True, null=True)
    position_in_document = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['position_in_document']
        verbose_name = 'Table'
        verbose_name_plural = 'Tables'
    
    def __str__(self):
        return f"Tabla {self.position_in_document} - {self.document.title}"


class Chunk(models.Model):
    """Document chunks for RAG retrieval."""
    
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    
    # Chunk data
    chunk_id = models.CharField(max_length=100)  # e.g., chunk_0000
    content = models.TextField()
    chunk_index = models.IntegerField()
    
    # Metadata from chunking
    metadata = models.JSONField(default=dict)
    
    # Embedding data
    embedding_vector = models.JSONField(null=True, blank=True)  # Stored as list
    embedding_dimension = models.IntegerField(null=True, blank=True)
    
    # ChromaDB reference
    chromadb_id = models.CharField(max_length=255, blank=True, null=True)
    indexed_in_chromadb = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['chunk_index']
        unique_together = ['document', 'chunk_id']
        verbose_name = 'Chunk'
        verbose_name_plural = 'Chunks'
    
    def __str__(self):
        return f"{self.document.title} - {self.chunk_id}"
    
    def get_preview(self, length=100):
        """Get a preview of the chunk content."""
        if len(self.content) <= length:
            return self.content
        return self.content[:length] + "..."


class ProcessingLog(models.Model):
    """Log entries for document processing steps."""
    
    LOG_LEVEL_CHOICES = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('success', 'Success'),
    ]
    
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='logs')
    level = models.CharField(max_length=10, choices=LOG_LEVEL_CHOICES, default='info')
    stage = models.CharField(max_length=50)  # parsing, chunking, embedding, indexing
    message = models.TextField()
    details = models.JSONField(null=True, blank=True)
    
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['created_at']
        verbose_name = 'Processing Log'
        verbose_name_plural = 'Processing Logs'
    
    def __str__(self):
        return f"[{self.level.upper()}] {self.stage} - {self.document.title}"
