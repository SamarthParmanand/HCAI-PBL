# from django.http import HttpResponse


# def index(request):
#     return HttpResponse("Hello, world. You're at the polls index.")

from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Samarth Parmanand", "matriculation": "672038"},
        {"name": "Meow Meow Meow", "matriculation": "000000"},
    ]
    
    projects = [
        {"name": "Home", "url_name": "home:index"},
        {"name": "Project 1", "url_name": "project1:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))