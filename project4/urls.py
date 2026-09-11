from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.index, name='index'),
    path('start/', views.start, name='start'),
    path('task/', views.task, name='task'),
    path('results/', views.results, name='results'),
    path('report/', views.download_report, name='report'),
    path('data/', views.download_data, name='data'),
]
