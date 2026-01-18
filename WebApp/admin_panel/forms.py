"""
Forms for admin_panel application.
"""
from django import forms
from .models import Document


class DocumentUploadForm(forms.ModelForm):
    """Form for uploading documents."""
    
    class Meta:
        model = Document
        fields = ['title', 'file']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Título del documento (opcional, se generará automáticamente)'
            }),
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.pdf,.docx,.doc,.txt,.md'
            }),
        }
        labels = {
            'title': 'Título',
            'file': 'Archivo',
        }
        help_texts = {
            'file': 'Formatos soportados: PDF, DOCX, DOC, TXT, MD',
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make title optional
        self.fields['title'].required = False
