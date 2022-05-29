from django.conf.urls import url
from . import main

urlpatterns = [
    url('videoinput', main.videoinput),
    url('showheartrate', main.showheartrate)
]