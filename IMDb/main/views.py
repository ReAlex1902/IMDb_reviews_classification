from .apps import MainConfig
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def index(request):
    return render(request, 'index.html')


def rating(request):
    sentence = request.POST['review'][0]
    return HttpResponse(MainConfig.get_rating(sentence))