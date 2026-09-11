from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('report/', views.download_report, name='report'),
    path('expert/', views.be_the_expert, name='expert'),
]
