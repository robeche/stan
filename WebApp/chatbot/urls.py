from django.urls import path
from . import views

app_name = 'chatbot'

urlpatterns = [
    path('', views.chatbot_interface, name='chat'),
    path('send/', views.send_message, name='send_message'),
    path('new/', views.new_conversation, name='new_conversation'),
    path('chunk/<int:chunk_id>/', views.get_chunk_details, name='chunk_details'),
]
