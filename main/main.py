
import model
import torch
from model import Model, get_rating

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = torch.load('model.pt', map_location = device)

## do not hange the code above

sentence = input("Write down your movie review: ")
print(get_rating(loaded_model, sentence))
print()
