## C:\Users\ASUS\AppData\Local\Programs\Python\Python36\Lib\site-packages

import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
import pickle
import spacy
import os

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers = n_layers,
                           bidirectional = bidirectional,
                           dropout = dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedding = self.embedding(text)    ## shape = (sent_length, batch_size)
        embedded = self.dropout(embedding)  ## shape = (sent_length, batch_size, emb_dim)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)    ## pack sequence

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)        ## unpack sequence

        ## output shape = (sent_len, batch_size, hid_dim * num_directions)
        ## output over padding tokens are zero tensors

        ## hidden shape = (num_layers * num_directions, batch_size, hid_dim)
        ## cell shape = (num_layers * num_directions, batch_size, hid_dim)

        ## concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        ## and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)) ## shape = (batch_size, hid_dim * num_directions)

        return self.fc(hidden)
current_directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
with open(os.path.join(current_directory, 'vocab.txt'), 'rb') as file:
    vocab = pickle.load(file)

nlp = spacy.load('en')

def predict_sentiment(model, sentence):
    '''
    inpur: model - AI model
           sentence - sentence to evaluate
    output - score between 0 and 1. 0 - negative, 1 - positive
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

def get_rating(model, sentence):
    '''
    inpur: model - AI model
           sentence - sentence to evaluate
    '''
    score = predict_sentiment(model, sentence)
    if score >= 0.7:
        class_ = 'Positive'
    elif 0.4 < score < 0.7:
        class_ = 'Medium'
    else:
        class_ = 'Negative'

    score = round(score, 1) * 10
    return {'rating': score, 'class': class_}

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = torch.load('model.pt', map_location = device)
    while True:
        # sentence = "Shut the fuck up those who say that it is the bad film. It's amazing"
        sentence = input("Write down your movie review: ")
        print(get_rating(loaded_model, sentence))
        print()
