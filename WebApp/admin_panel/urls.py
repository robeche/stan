"""
URL configuration for admin_panel app.
"""
from django.urls import path
from . import views

app_name = 'admin_panel'

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    path('model-diagram/', views.model_diagram, name='model_diagram'),
    
    # Document management
    path('documents/', views.document_list, name='document_list'),
    path('documents/upload/', views.document_upload, name='document_upload'),
    path('documents/<int:pk>/', views.document_detail, name='document_detail'),
    path('documents/<int:pk>/delete/', views.document_delete, name='document_delete'),
    path('documents/<int:pk>/reprocess/', views.document_reprocess, name='document_reprocess'),
    
    # Processing status API
    path('api/task-status/<str:task_id>/', views.task_status_api, name='task_status_api'),
    path('api/document-status/<int:document_id>/', views.document_status_api, name='document_status_api'),
]
