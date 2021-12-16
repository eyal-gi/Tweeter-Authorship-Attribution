import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # RelU, tanh
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import ex3_307887984_307830901 as ex3
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    def __init__(self, input_size, hidden_size, num_classes, dropout_p):
        super(NN, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)
        # Activation, dropout, batch-normalization layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
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

    def fit(self, train_loader, criterion, optimizer, epochs, validation_loader=None):
        history = {'accuracy': [0],
                   'val_accuracy': [0],
                   'loss': [1],
                   'val_loss': [1]
                   }
        # training loop
        self.train()
        for epoch in range(1, epochs + 1):
            epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc = 0, 0, 0, 0
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
            if validation_loader:
                for X_batch, y_batch in validation_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    y_pred = self(X_batch)
                    loss = criterion(y_pred, y_batch.unsqueeze(1))
                    acc = self._binary_acc(y_pred, y_batch.unsqueeze(1))

                    val_epoch_loss += loss.item()
                    val_epoch_acc += acc.item()

            history['accuracy'].append(epoch_acc / len(train_loader))
            history['loss'].append(epoch_loss / len(train_loader))
            history['val_accuracy'].append(val_epoch_acc / len(validation_loader))
            history['val_loss'].append(val_epoch_loss / len(validation_loader))
            print(
                f'Epoch {epoch}/{epochs}\n[=================] - loss: {epoch_loss / len(train_loader):.5f} - accuracy: {epoch_acc / len(train_loader):.4f} - val_loss: {val_epoch_loss / len(validation_loader):.5f} - val_accuracy: {val_epoch_acc / len(validation_loader):.4f}')

        return history

    def _binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        # acc = torch.round(acc * 100)

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
        acc = (y_pred_list == y_test).sum().item() / len(test_loader.dataset)
        return acc
        # print(f'accuracy = {acc:.3f}')

    def plot_acc_loss(self, history_dict):
        # accuracy plot
        plt.plot(history_dict['accuracy'])
        plt.plot(history_dict['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        # loss plot
        plt.plot(history_dict['loss'])
        plt.plot(history_dict['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


# # ###########____________________________________________________________________###################################

# Model's Parameters
input_size_ = 15
hidden_size_ = 100
num_classes_ = 1
num_epochs_ = 32
batch_size_ = 16
learning_rate_ = 0.001

# Load Datasets
X_train, Y_train, X_test = ex3.read_and_split_data()


def ann_tuning(X, y, params):
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    options = 1
    for p in params:
        options = options * len(p)
    print(f'{options} Options -> {10 * options} iterations')

    tuning_params = []
    tuning_train_acc = []
    tuning_val_acc = []

    for i_s in params['INPUT_SIZE']:
        for h_s in params['HIDDEN_SIZE']:
            for c in params['CLASSES']:
                for e in params['EPOCHS']:
                    for b_s in params['BATCH_SIZE']:
                        for lr in params['LR']:
                            for d_p in params['DROPOUT_P']:
                                print(
                                    f'This tuning parametrs are:\n input_size:{i_s}, hidden_size:{h_s}, epochs:{e}, batch_size:{b_s}, learning_rate:{lr}, dropout_p:{d_p}'
                                )
                                iter_params = {'input_size': i_s, 'hidden_size': h_s, 'classes': c, 'epochs': e,
                                               'batch_size': b_s, 'learning_rate': lr, 'dropout_p': d_p}
                                nn_clf = NN(input_size=i_s, hidden_size=h_s, num_classes=c, dropout_p=d_p).to(device)
                                for train_index, test_index in skf.split(X, y):
                                    print("TRAIN:", train_index, "TEST:", test_index)
                                    x_train, x_val = X[train_index], X[test_index]
                                    y_train, y_val = y[train_index], y[test_index]

def prepare_datasets(x_train, y_train, x_validation, y_validation):
    x_train = x_train.to_numpy()
    x_validation = x_validation.to_numpy()
    y_train = y_train.to_numpy()
    y_validation = y_validation.to_numpy()

    #  Convert datasets
    train_dataset = ConvertDataset(x=torch.FloatTensor(x_train), y=torch.FloatTensor(y_train), train=True)
    validation_dataset = ConvertDataset(x=torch.FloatTensor(x_validation), y=torch.FloatTensor(y_validation),
                                        train=True)

params_grid = {'INPUT_SIZE': [X_train.shape[1]],
               'HIDDEN_SIZE': [16, 32, 64, 128, 256],
               'CLASSES': [1],
               'EPOCHS': [8, 16, 32, 64],
               'BATCH_SIZE': [16, 32, 64, 128],
               'LR': [0.001, 0.01, 0.1],
               'DROPOUT_P': [0.1, 0.3, 0.5]
               }

ann_tuning(X=X_train, y=Y_train, params=params_grid)

# ####################################################################################################################
# ####################################################################################################################
# ####################################################################################################################
# ####################################################################################################################
# # ###########____________________________________________________________________###################################
x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

x_train = x_train.to_numpy()
x_validation = x_validation.to_numpy()
y_train = y_train.to_numpy()
y_validation = y_validation.to_numpy()

#  Convert datasets
train_dataset = ConvertDataset(x=torch.FloatTensor(x_train), y=torch.FloatTensor(y_train), train=True)
validation_dataset = ConvertDataset(x=torch.FloatTensor(x_validation), y=torch.FloatTensor(y_validation), train=True)
# test_dataset = ConvertDataset(x=X_test.values)

# Create DataLoaders
train_loader_ = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=True)
validation_loader_ = DataLoader(dataset=validation_dataset, batch_size=batch_size_, shuffle=True)
# # todo: test_loader?
# # ###########____________________________________________________________________###################################
#
#
# # ###########____________________________________________________________________###################################
# Initiate model
nn_model = NN(input_size=input_size_, hidden_size=hidden_size_, num_classes=num_classes_).to(device)

# loss and optimizer
criterion_ = nn.BCEWithLogitsLoss()
optimizer_ = optim.Adam(nn_model.parameters(), lr=learning_rate_)

history = nn_model.fit(train_loader=train_loader_,
                       criterion=criterion_,
                       optimizer=optimizer_, epochs=num_epochs_,
                       validation_loader=validation_loader_)

# nn_model.evaluate(test_loader=validation_loader_, y_test=y_validation)
#
#
# ########### report ###############
# # print(confusion_matrix(y_validation, y_pred_list))
# # print(classification_report(y_validation, y_pred_list))
# #         n_samples += labels.shape[0]
# #         n_correct += (outputs == labels).sum().item()
# #
# # ###########____________________________________________________________________###################################
