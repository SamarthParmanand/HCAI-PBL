from django.http import HttpResponse
from django.template import loader

# The group, in one place: the home page lists it and the Project 2 report cites
# it, so the two can never drift apart.
STUDENTS = [
    {"name": "Samarth Parmanand", "matriculation": "672038"},
]

PROJECTS = [
    {"name": "Project 1", "url_name": "project1:index",
     "summary": "Supervised learning interface: upload a dataset, explore it, train a model."},
    {"name": "Project 2", "url_name": "project2:index",
     "summary": "Explainability: complexity trade-off, counterfactuals, PDP and ALE."},
    {"name": "Project 3", "url_name": "project3:index",
     "summary": "Learning-to-defer on AG News, and active learning for expert competence."},
    {"name": "Project 4", "url_name": "project4:index",
     "summary": "Preference elicitation: a movie recommender and a user-study interface."},
]


def index(request):
    template = loader.get_template("home/index.html")

    context = {
        "students": STUDENTS,
        "projects": PROJECTS,
    }

    return HttpResponse(template.render(context, request))
