
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_breast_cancer



def read_and_split_data():
    """
    This function read the data after features selection and split it to X_train , Y_train and for X_test
    """
    train_df = pd.read_csv('train_df.csv')
    X_train = train_df.drop('label', axis=1)
    Y_train = train_df['label']

    return X_train, Y_train



# ___

# class dataset(Dataset):
#     def __init__(self, x, y):
#         self.x = torch.tensor(x, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
#         self.length = self.x.shape[0]
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
#
#     def __len__(self):
#         return self.length
#     #########################################################
#
#     def __init__(self, x, y=None):
#         self.x = x
#         self.y = y
#         self.length = self.x.shape[0]
#
#
#     def __getitem__(self, idx):
#         x_idx = torch.tensor(self.x[idx], dtype=torch.float32)
#         if self.y is not None:
#             y_idx = torch.tensor(self.y[idx], dtype=torch.float32)
#         else:
#             y_idx = None
#         return x_idx, y_idx
#     def __len__(self):
#         return self.length

class train_dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length

class test_dataset(Dataset):
        def __init__(self, x):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.length = self.x.shape[0]

        def __getitem__(self, idx):
            return self.x[idx]

        def __len__(self):
            return self.length



class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,32)
    self.fc2 = nn.Linear(32,64)
    self.fc3 = nn.Linear(64,1)
  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x



def train_NN():
    x, y = read_and_split_data()
    sc = StandardScaler()
    x = sc.fit_transform(x)
    # hyper parameters
    learning_rate = 1
    epochs = 700
    params = {'lr': [0.01],
              'batch_size': [100]}
    mean_metric_results = {'learning_rate': [], 'score': []}
    param_values = [v for v in params.values()]
    # Optimizers require the parameters to optimize and a learning rate

    for lr, batch_size in product(*param_values):
        X_train, X_val, Y_train, Y_val = train_test_split(x,y, test_size=0.1,random_state=42)


        trainset = train_dataset(X_train, Y_train)
        valset = test_dataset(X_val)
        # DataLoader
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
         # Model , Optimizer, Loss
        model = Net(input_shape=x.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        train_losses = []
        train_accur = []
        val_losses = []
        val_accur = []
        for i in range(epochs):
            for j, (x_train, y_train) in enumerate(trainloader):
                # calculate output
                output = model(x_train)
                # calculate loss
                loss = loss_fn(output, y_train.reshape(-1, 1))
                # accuracy
                predicted = model(torch.tensor(X_train, dtype=torch.float32))
                # print((predicted.reshape(-1).detach().numpy().round()))
                # print (y_train)
                # print (predicted.reshape(-1).detach().numpy().round())
                # print (y)
                acc = (predicted.reshape(-1).detach().numpy().round() == Y_train).mean()
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i % 50 == 0:
                train_losses.append(loss)
                train_accur.append(acc)
                print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))
            for j, (x_val) in enumerate(valloader):
                # calculate output
                output = model(x_val)
                # calculate loss
                # accuracy
                predicted = model(torch.tensor(X_val, dtype=torch.float32))
                acc = (predicted.reshape(-1).detach().numpy().round() == Y_val).mean()
                # backprop

            if i % 50 == 0:
                val_losses.append(loss)
                val_accur.append(acc)
                print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))
    # fo i in range(len(val_losses)):



train_NN()

