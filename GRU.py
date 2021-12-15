import pandas as pd
import torch
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib as plt


class GRU(nn.Module):
    def __init__(self, vocab_size, batch_size, embedding_dimension=100, hidden_size=128, n_layers=1, device='cpu'):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = batch_size

        # self.encoder = nn.Embedding(10, 50)  # change value?
        self.encoder = nn.Embedding(vocab_size, embedding_dimension)
        # self.rnn = nn.GRU(50, 128)  # change value?
        self.rnn = nn.GRU(
            input_size=embedding_dimension,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True)
        self.decoder = nn.Linear(hidden_size, 1)

    def init_hidden(self):
        return torch.randn(1, 1, 128)

    def forward(self, input_, hidden):
        encoded = self.encoder(input_)
        output, hidden = self.rnn(encoded.unsqueeze(1), hidden)
        output = self.decoder(output.squeeze(1))
        return output, hidden
