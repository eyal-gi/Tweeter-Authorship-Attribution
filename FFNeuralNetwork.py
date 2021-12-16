import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # RelU, tanh
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import ex3_307887984_307830901 as ex3
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class ConvertDataset(Dataset):
    def __init__(self, x, y=None, train=False):
        # data loading
        self.train = train

        self.x = x
        if train: self.y = y

    def __getitem__(self, index):
        if self.train:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

    def __len__(self):
        return len(self.x)


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)
        # Activation, dropout, batch-normalization layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        out = self.fc_out(x)

        return out

    def fit(self, train_loader, criterion, optimizer, epochs, validation_loader):
        # training loop
        self.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                # forward
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = self._binary_acc(y_pred, y_batch.unsqueeze(1))

                # backwards
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            # print(f'Epoch {epoch}/{epochs}\n {train_loader.__len__()/epochs}/{train_loader.__len__()/epochs} [=================] - ')
            print(
                f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

        # if (i + 1) % 10 == 0:
        #     pass
        #     print(f'Epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

    def _binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def evaluate(self, test_loader, y_test):
        # evaluation
        y_pred_list = []
        self.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(device)
                y_test_pred = self(X_batch)
                y_test_pred = torch.sigmoid(y_test_pred)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        acc = (y_pred_list == y_test).sum().item() / test_loader.__len__()
        return acc
        # print(f'accuracy = {acc:.3f}')


# # ###########____________________________________________________________________###################################

# Model's Parameters
input_size_ = 15
hidden_size_ = 100
num_classes_ = 1
num_epochs_ = 32
batch_size_ = 100
learning_rate_ = 0.001

# Load Datasets
X_train, Y_train, X_test = ex3.read_and_split_data()
# train_df = pd.read_csv('train_df.csv')
# # X_train = train_df.iloc[:, 1:]
# # Y_train = train_df.iloc[:, 0]

# # ###########____________________________________________________________________###################################
x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

x_train = x_train.to_numpy()
x_validation = x_validation.to_numpy()
y_train = y_train.to_numpy()
y_validation = y_validation.to_numpy()

#  Convert datasets
train_dataset = ConvertDataset(x=torch.FloatTensor(x_train), y=torch.FloatTensor(y_train), train=True)
validation_dataset = ConvertDataset(x=torch.FloatTensor(x_validation), train=False)
# test_dataset = ConvertDataset(x=X_test.values)

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)
# # todo: test_loader?
# # ###########____________________________________________________________________###################################
#
#
# # ###########____________________________________________________________________###################################
# Initiate model
nn_model = NN(input_size=input_size_, hidden_size=hidden_size_, num_classes=num_classes_).to(device)
print(nn_model)

# loss and optimizer
criterion_ = nn.BCEWithLogitsLoss()
optimizer_ = optim.Adam(nn_model.parameters(), lr=learning_rate_)

nn_model.fit(train_loader=train_loader, criterion=criterion_, optimizer=optimizer_, epochs=num_epochs_)
nn_model.evaluate(test_loader=validation_loader, y_test=y_validation)
#
#
# ########### report ###############
# # print(confusion_matrix(y_validation, y_pred_list))
# # print(classification_report(y_validation, y_pred_list))
# #         n_samples += labels.shape[0]
# #         n_correct += (outputs == labels).sum().item()
# #
# # ###########____________________________________________________________________###################################


# load data
X_train, Y_train, X_test = ex3.read_and_split_data()
