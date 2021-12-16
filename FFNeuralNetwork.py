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

        # self.x = torch.tensor(x, dtype=torch.float32)
        self.x = x
        if train:
            # self.y = torch.tensor(y, dtype=torch.float32)
            self.y = y

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
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        out = self.fc_out(x)

        return out

        # out = self.fc1(x)
        # out = self.relu1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        # out = self.fc3(out)
        # out = self.softmax(out)
        # return out


# Model's Parameters
input_size = 15
hidden_size = 100
num_classes = 1
num_epochs = 32
batch_size = 100
learning_rate = 0.001

# Load Datasets
# X_train, Y_train, X_test = ex3.read_and_split_data()
train_df = pd.read_csv('train_df.csv')
X_train = train_df.iloc[:, 1:]
Y_train = train_df.iloc[:, 0]

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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=1)
# todo: test_loader?


# Initiate model
model = NN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
print(model)

# loss and optimizer
# criterion = nn.BCELoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


# training loop
model.train()
n_total_steps = len(train_loader)
for epoch in range(1, num_epochs + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        # backwards
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(
        f'Epoch {epoch + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    # if (i + 1) % 10 == 0:
    #     pass
    #     print(f'Epoch {epoch + 1} / {num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

# evaluation
y_pred_list = []
model.eval()
with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
    for X_batch in validation_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

print(confusion_matrix(y_validation, y_pred_list))
print(classification_report(y_validation, y_pred_list))
#         n_samples += labels.shape[0]
#         n_correct += (outputs == labels).sum().item()
#
#     acc = n_correct / n_samples
#     print(f'accuracy = {acc:.3f}')
