from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.index, name='index'),
    path('explore/', views.explore, name='explore'),
    path('train/', views.train, name='train'),
    path('download/', views.download, name='download'),
]
