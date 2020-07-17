# Analysis and model creation
Go to analysis.ipynb to see the analysis, model creation and metrics. The accuracy is ~88% as well as F1 score.

# Scripts
model.py - NN model with necessary methods\
model.pt - saved NN model\
vocab.txt - vocabulary

### Use model.py, model.pt and vocab.txt for using the Neural Net in your products. The example of using is in main.py

# Start the application
To start the application go to IMDb folder and run manage.py script. To start the application you need next libraries:
1) PyTorch (https://pytorch.org/)
2) Spacy with English (https://spacy.io/usage/models)
3) TorchText version 0.3.1 (pip install torchtext==0.3.1)
4) Pickle (pip install pickle)
