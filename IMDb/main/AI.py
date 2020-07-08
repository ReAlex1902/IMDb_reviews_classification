import os
import json
import torch
from .model import Model, get_rating

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__current_directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
loaded_model = torch.load(os.path.join(__current_directory, 'model.pt'), map_location = device)

## do not hange the code above
def get(sentence):
    result = get_rating(loaded_model, sentence)
    return json.dumps(result)