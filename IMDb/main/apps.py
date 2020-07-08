import os
import json
from django.apps import AppConfig
import torch
from . import AI

class MainConfig(AppConfig):
    name = 'main'
    @staticmethod
    def get_rating(sentence):
        return AI.get(sentence)
