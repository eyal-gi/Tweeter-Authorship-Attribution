import string
import pickle
from nltk.corpus import stopwords
from tabulate import tabulate
from itertools import product
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
import re
import nltk
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')

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
    """
    Creates an instance of a Multi Layer Perceptron.
    Inner methods: model fit, model evaluation.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initiates the neural network model.
        :param input_size (int): The input layer size (number of features / embedding vector length)
        :param hidden_size (list): List on integers defining the hidden layers sizes (number of neurons in each layer)
        :param num_classes (int): The number of classification classes
        """
        super(NN, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        # self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        # self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        self.fc_out = nn.Linear(hidden_size[1], num_classes)
        # Activation, dropout, batch-normalization layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size[0])
        self.batchnorm2 = nn.BatchNorm1d(hidden_size[1])

    def forward(self, inputs):
        """
        Defines the model's architecture.

        :param inputs: (list) Input layer
        :return: predictions
        """
        x = self.relu(self.fc1(inputs))
        # x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        x = self.fc_out(x)
        out = torch.sigmoid(x)

        return out

    def fit(self, train_loader, criterion, optimizer, epochs, validation_loader=None, verbose=0):
        """
        Fits the model to a given train dataset.
        :param train_loader: Instance of type torch.DataLoader of the train data (with labels)
        :param criterion: The network criterion
        :param optimizer: The network optimizer
        :param epochs: Number of training epochs
        :param validation_loader: Instance of type torch.DataLoader of the validation data (with labels)
        :param verbose: Defauls=0. If verbose=1, the program will print information during training.
        :return: history. a Dictionary containing the train and validation loss and accuracy during training.
        """
        history = {'accuracy': [],
                   'val_accuracy': [],
                   'loss': [],
                   'val_loss': []
                   }
        self.train()  # model.train() indicates the model this is model training

        # run through all epochs
        for epoch in range(1, epochs + 1):
            # initiate train, validation loss and accuracy for each epoch
            epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc = 0, 0, 0, 0
            # run through the batches
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()  # zero the optimizer
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # send to cpu/gpu

                # forward
                y_pred = self(X_batch)  # predict based on X
                loss = criterion(y_pred, y_batch.unsqueeze(1))  # loss based on predicted vs ground truth
                acc = self._binary_acc(y_pred, y_batch.unsqueeze(1))

                # backwards
                loss.backward()
                optimizer.step()

                # sum loss and accuracy for every batch
                epoch_loss += loss.item()
                epoch_acc += acc.item()

            # if trained with a validation set:
            if validation_loader:
                for X_batch, y_batch in validation_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    y_pred = self(X_batch)
                    loss = criterion(y_pred, y_batch.unsqueeze(1))
                    acc = self._binary_acc(y_pred, y_batch.unsqueeze(1))

                    val_epoch_loss += loss.item()
                    val_epoch_acc += acc.item()

                history['val_accuracy'].append(val_epoch_acc / len(validation_loader))
                history['val_loss'].append(val_epoch_loss / len(validation_loader))

            history['accuracy'].append(epoch_acc / len(train_loader))
            history['loss'].append(epoch_loss / len(train_loader))

            if verbose == 1:
                if validation_loader:
                    print(
                        f'Epoch {epoch}/{epochs}\n[=================] - loss: {epoch_loss / len(train_loader):.5f} - accuracy: {epoch_acc / len(train_loader):.4f} - val_loss: {val_epoch_loss / len(validation_loader):.5f} - val_accuracy: {val_epoch_acc / len(validation_loader):.4f}')
                else:
                    print(
                        f'Epoch {epoch}/{epochs}\n[=================] - loss: {epoch_loss / len(train_loader):.5f} - accuracy: {epoch_acc / len(train_loader):.4f}')
        return history

    def _binary_acc(self, y_pred, y_test):
        """
        Returns accuracy for binary classification
        :param y_pred: model predictions
        :param y_test: Ground truth (list)
        :return: float -> the accuracy
        """
        y_pred_tag = torch.round(y_pred)
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]

        return acc

    def evaluate(self, x_train, x_test, y_train, y_test):
        """
        Evaluate the model on a test set and returns the train and test accuracy.
        :param x_train: np.array of train data
        :param x_test: np.array of train labels
        :param y_train: np.array of test data
        :param y_test: np.array of test labels
        :return:
        """
        # evaluation
        self.eval()  # model.eval() indicates the model this is model eval
        train_predicted = self(torch.tensor(x_train, dtype=torch.float32))
        # train_predicted = train_predicted
        train_acc = (train_predicted.reshape(-1).detach().numpy().round() == y_train).mean()

        test_predicted = self(torch.tensor(x_test, dtype=torch.float32))
        # test_predicted = test_predicted
        test_acc = (test_predicted.reshape(-1).detach().numpy().round() == y_test).mean()

        return train_acc, test_acc

    def plot_acc_loss(self, history_dict):
        """
        Plots the accuracy and loss over training
        :param history_dict: model training history
        """
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


def ann(X, y, X_val, y_val, input_size, hidden_size, batch_size, lr, epochs):
    """
    construct a NeuralNetwork model.

    :param X: train dataframe
    :param y: train labels dataframe
    :param input_size: input layer size
    :param hidden_size: hidden layers size list
    :param batch_size: training batches size
    :param lr: solver's learning rate
    :param epochs: number of training epochs
    :return: ANN model
    """
    # Convert data sets
    train_set, valid_set = prepare_datasets(X, y, X_val, y_val, batch_size)
    # define model, model criterion and optimizer
    nn_clf = NN(input_size=input_size, hidden_size=hidden_size, num_classes=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(nn_clf.parameters(), lr=lr)
    history = nn_clf.fit(train_loader=train_set,
                         validation_loader=valid_set,
                         criterion=criterion,
                         optimizer=optimizer,
                         epochs=epochs, verbose=1)

    return nn_clf, history


def prepare_datasets(x_train, y_train, x_validation, y_validation, batch_size):
    """
    Converts DataFrames to DataLoaders
    :param x_train: train dataframe
    :param y_train: train labels dataframe
    :param x_validation: test/validation dataframe
    :param y_validation: validation labels dataframe
    :param batch_size: Batch size (int)
    :return: train and test/validation DataLoaders
    """

    # convert dataframe to numpy arrays
    x_train = x_train.to_numpy()
    x_validation = x_validation.to_numpy()
    y_train = y_train.to_numpy()
    y_validation = y_validation.to_numpy()

    #  Convert to torch Datasets
    train_dataset = ConvertDataset(x=torch.FloatTensor(x_train),
                                   y=torch.FloatTensor(y_train),
                                   train=True)
    validation_dataset = ConvertDataset(x=torch.FloatTensor(x_validation),
                                        y=torch.FloatTensor(y_validation),
                                        train=True)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader


def kfold_tuning(X, y, params):
    """
    Performs a 10-Fold validation over given parameters and returns a DataFrame of the results.
    :param X: Train samples (DataFrame)
    :param y: Train labels (DataFrame)
    :param params: Dictionary of parameters.
    :return: Results data frame
    """
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    # initialize results lists
    tuning_params = []
    tuning_train_acc = []
    tuning_val_acc = []

    # counts number of tuning options
    options = 1
    for p in params:
        options = options * len(params[p])
    print(f'{options} Options -> {10 * options} iterations')

    # loop through all possible combinations
    param_values = [v for v in params.values()]
    i = 1  # index of current iteration
    for i_s, h_s, e, b_s, lr in product(*param_values):
        print(
            f'{i}/{options} Tuning parameters-> input_size:{i_s} | hidden_size:{h_s} | epochs:{e} | batch_size:{b_s} | learning_rate:{lr}'
        )
        i += 1

        # initiates cross-validation results
        cv_train_acc = []
        cv_val_acc = []
        # loop through the folds
        for train_index, test_index in skf.split(X, y):
            x_train, x_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            # initiate the neural network
            nn_clf, history = ann(X=x_train, y=y_train, X_val=x_val, y_val=y_val, input_size=i_s, hidden_size=h_s,
                                  batch_size=b_s, lr=lr, epochs=e)
            # evaluate on the validation
            acc = nn_clf.evaluate(x_train.to_numpy(), x_val.to_numpy(), y_train.to_numpy(),
                                  y_val.to_numpy())
            # append results of the fold
            cv_train_acc.append(acc[0])
            cv_val_acc.append(acc[1])
            nn_clf.plot_acc_loss(history)
        print(f'train_acc: {np.mean(cv_train_acc):.3f}, val_acc:{np.mean(cv_val_acc):.3f}')
        iter_params = {'input_size': i_s, 'hidden_size': h_s, 'epochs': e,
                       'batch_size': b_s, 'learning_rate': lr}
        # append results of the iteration
        tuning_params.append(iter_params)
        tuning_val_acc.append(np.mean(cv_val_acc))
        tuning_train_acc.append(np.mean(cv_train_acc))

    cv_results = {'params': tuning_params, 'mean_test_score': tuning_val_acc, 'mean_train_score': tuning_train_acc}
    return cv_results


def ann_tuning(x_train, y_train, params_grid):
    """
    Preforms parameters tuning on the ann
    :param x_train: train data frame
    :param y_train: train labels data frame
    :param params_grid: tuning parameters (dictionary)
    :return: Best model
    """
    results = kfold_tuning(X=x_train, y=y_train, params=params_grid)
    # convert dictionary to DataFrame
    results = pd.DataFrame(results).sort_values('mean_test_score', ascending=False)
    # print table
    headers = ['Parameters', 'Validation score', 'Train score']
    print(tabulate(results.head(10), headers=headers, tablefmt='grid'))

    return results.head(1)


# Load Datasets
X_train, Y_train, X_test = ex3.read_and_split_data()

params_grid_3layers = {'INPUT_SIZE': [X_train.shape[1]],
                       'HIDDEN_SIZE': [[32, 32, 32, 32], [64, 64, 64, 64], [128, 128, 128, 128], [128, 128, 64, 64],
                                       [128, 64, 128, 64],
                                       [128, 64, 32, 16], [64, 64, 64, 32], [16, 32, 64, 128], [32, 32, 64, 64],
                                       [64, 64, 128, 128]],
                       'EPOCHS': [8, 16, 32, 64],
                       'BATCH_SIZE': [16, 32, 64, 128],
                       'LR': [0.001, 0.01]
                       }

params_grid_3layers = {'INPUT_SIZE': [X_train.shape[1]],
                       'HIDDEN_SIZE': [[128, 128, 128, 128], [256, 256, 256, 256]],
                       'EPOCHS': [64, 128, 256],
                       'BATCH_SIZE': [64],
                       'LR': [0.001]
                       }

params_grid_4layers = {'INPUT_SIZE': [X_train.shape[1]],
                       'HIDDEN_SIZE': [[128, 128, 128, 128, 128], [256, 256, 256, 256, 256]],
                       'EPOCHS': [64, 128, 256],
                       'BATCH_SIZE': [64],
                       'LR': [0.001]
                       }
params_grid_1layer = {'INPUT_SIZE': [X_train.shape[1]],
                      'HIDDEN_SIZE': [[16, 16], [32, 32], [64, 64], [128, 128], [16, 8], [32, 16], [64, 32], [128, 64]],
                      'EPOCHS': [4, 8, 16, 32, 64],
                      'BATCH_SIZE': [16, 32, 64, 128],
                      'LR': [0.001, 0.01]
                      }

test_params = {'INPUT_SIZE': [X_train.shape[1]],
               'HIDDEN_SIZE': [[128, 128]],
               'EPOCHS': [100],
               'BATCH_SIZE': [64],
               'LR': [0.001]
               }

# best_model = ann_tuning(x_train=X_train, y_train=Y_train, params_grid=test_params)

X, x, Y, y = train_test_split(X_train, Y_train, train_size=0.8, random_state=42, stratify=Y_train)
model, history = ann(X=X, y=Y, X_val=x, y_val=y, input_size=15, hidden_size=(128, 128), batch_size=64, lr=0.001,
                     epochs=100)
print(history)
model.plot_acc_loss(history)


