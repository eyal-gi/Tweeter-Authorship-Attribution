from tabulate import tabulate
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
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc_out = nn.Linear(hidden_size[1], num_classes)
        # Activation, dropout, batch-normalization layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size[0])
        self.batchnorm2 = nn.BatchNorm1d(hidden_size[1])

    def forward(self, inputs):
        x = self.relu(self.fc1(inputs))
        # x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        out = self.fc_out(x)

        return out

    def fit(self, train_loader, criterion, optimizer, epochs, validation_loader=None, verbose=0):
        history = {'accuracy': [0],
                   'val_accuracy': [0],
                   'loss': [1],
                   'val_loss': [1]
                   }
        # training loop
        self.train()
        # run through all epochs
        for epoch in range(1, epochs + 1):
            # initiate train, validation loss and accuracy for each epoch
            epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc = 0, 0, 0, 0
            # run through the batches
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # forward
                y_pred = self(X_batch)  # predict based on X
                loss = criterion(y_pred, y_batch.unsqueeze(1))  # loss based on predicted vs ground truth
                acc = self._binary_acc(y_pred, y_batch.unsqueeze(1))

                optimizer.zero_grad()
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

                history['val_accuracy'].append(val_epoch_acc / len(validation_loader))
                history['val_loss'].append(val_epoch_loss / len(validation_loader))

            history['accuracy'].append(epoch_acc / len(train_loader))
            history['loss'].append(epoch_loss / len(train_loader))

            if verbose == 1:
                print(
                    f'Epoch {epoch}/{epochs}\n[=================] - loss: {epoch_loss / len(train_loader):.5f} - accuracy: {epoch_acc / len(train_loader):.4f} - val_loss: {val_epoch_loss / len(validation_loader):.5f} - val_accuracy: {val_epoch_acc / len(validation_loader):.4f}')

        return history

    def _binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        # acc = torch.round(acc * 100)

        return acc

    def evaluate(self, x_train, x_test, y_train, y_test):
        # evaluation
        self.eval()
        train_predicted = self(torch.tensor(x_train, dtype=torch.float32))
        train_predicted = torch.sigmoid(train_predicted)
        train_acc = (train_predicted.reshape(-1).detach().numpy().round() == y_train).mean()

        test_predicted = self(torch.tensor(x_test, dtype=torch.float32))
        test_predicted = torch.sigmoid(test_predicted)
        test_acc = (test_predicted.reshape(-1).detach().numpy().round() == y_test).mean()

        # print(f'train accuracy = {train_acc:.3f}, test accuracy = {test_acc:.3f}')

        return train_acc, test_acc

    # def evaluate(self, train_loader, test_loader, y_test):
    #     # evaluation
    #     y_pred_list = []
    #     train_y_pred_list = []
    #     self.eval()
    #     with torch.no_grad():
    #         for X_batch in test_loader:
    #             X_batch = X_batch.to(device)
    #             y_test_pred = self(X_batch)
    #             y_test_pred = torch.sigmoid(y_test_pred)
    #             y_pred_tag = torch.round(y_test_pred)
    #             y_pred_list.append(y_pred_tag.cpu().numpy())
    #
    #     with torch.no_grad():
    #         for X_batch, y_batch in train_loader:
    #             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    #             y_train_pred = self(X_batch)
    #             train_acc = self._binary_acc(y_train_pred, y_batch.unsqueeze(1))
    #
    #     y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    #     test_acc = (y_pred_list == y_test).sum().item() / len(test_loader.dataset)
    #
    #     print(f'train accuracy = {train_acc:.3f}, test accuracy = {test_acc:.3f}')
    #
    #     return train_acc, test_acc

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
hidden_size_ = [62, 32]
num_classes_ = 1
num_epochs_ = 8
batch_size_ = 16
learning_rate_ = 0.001

# Load Datasets
X_train, Y_train, X_test = ex3.read_and_split_data()


def ann_tuning(X, y, params):
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    tuning_params = []
    tuning_train_acc = []
    tuning_val_acc = []


    options = 1
    for p in params:
        options = options * len(params[p])
    print(f'{options} Options -> {10 * options} iterations')


    for i_s in params['INPUT_SIZE']:
        for h_s in params['HIDDEN_SIZE']:
            for c in params['CLASSES']:
                for e in params['EPOCHS']:
                    for b_s in params['BATCH_SIZE']:
                        for lr in params['LR']:
                            for d_p in params['DROPOUT_P']:
                                print(
                                    f'Tuning parameters-> input_size:{i_s} | hidden_size:{h_s} | epochs:{e} | batch_size:{b_s} | learning_rate:{lr} | dropout_p:{d_p}'
                                )

                                nn_clf = NN(input_size=i_s, hidden_size=h_s, num_classes=c, dropout_p=d_p).to(device)
                                cv_train_acc = []
                                cv_train_loss = []
                                cv_val_acc = []
                                cv_val_loss = []
                                for train_index, test_index in skf.split(X, y):
                                    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
                                    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
                                    train_set, val_set = prepare_datasets(x_train, y_train, x_val, b_s)

                                    criterion = nn.BCEWithLogitsLoss()
                                    optimizer = optim.Adam(nn_clf.parameters(), lr=lr)

                                    nn_clf.fit(train_loader=train_set,
                                               criterion=criterion,
                                               optimizer=optimizer,
                                               epochs=e)

                                    acc = nn_clf.evaluate(x_train.to_numpy(), x_val.to_numpy(), y_train.to_numpy(),
                                                          y_val.to_numpy())
                                    cv_train_acc.append(acc[0])
                                    cv_val_acc.append(acc[1])
                                print(f'train_acc: {np.mean(cv_train_acc):.3f}, val_acc:{np.mean(cv_val_acc):.3f}')
                                iter_params = {'input_size': i_s, 'hidden_size': h_s, 'classes': c, 'epochs': e,
                                               'batch_size': b_s, 'learning_rate': lr, 'dropout_p': d_p}
                                tuning_params.append(iter_params)
                                tuning_val_acc.append(np.mean(cv_val_acc))
                                tuning_train_acc.append(np.mean(cv_train_acc))

    cv_results = {'params': tuning_params, 'mean_test_score': tuning_val_acc, 'mean_train_score': tuning_train_acc}
    return cv_results

def prepare_datasets(x_train, y_train, x_validation, batch_size):
    x_train = x_train.to_numpy()
    x_validation = x_validation.to_numpy()
    y_train = y_train.to_numpy()

    #  Convert datasets
    train_dataset = ConvertDataset(x=torch.FloatTensor(x_train),
                                   y=torch.FloatTensor(y_train),
                                   train=True)
    validation_dataset = ConvertDataset(x=torch.FloatTensor(x_validation),
                                        train=False)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=True)

    return train_loader, validation_loader


params_grid = {'INPUT_SIZE': [X_train.shape[1]],
               'HIDDEN_SIZE': [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256]],
               'CLASSES': [1],
               'EPOCHS': [8, 16, 32, 64],
               'BATCH_SIZE': [16, 32, 64, 128],
               'LR': [0.001, 0.01, 0.1],
               'DROPOUT_P': [0.1, ]
               }

params_grid = {'INPUT_SIZE': [X_train.shape[1]],
               'HIDDEN_SIZE': [[16, 16], [32, 32]],
               'CLASSES': [1],
               'EPOCHS': [8, 16],
               'BATCH_SIZE': [128],
               'LR': [0.1],
               'DROPOUT_P': [0.1]
               }

results = ann_tuning(X=X_train, y=Y_train, params=params_grid)
results = pd.DataFrame(results).sort_values('mean_test_score', ascending=False)
headers = ['Parameters', 'Validation score', 'Train score']
print(tabulate(results.head(), headers=headers, tablefmt='grid'))

# ####################################################################################################################
# ####################################################################################################################
# ####################################################################################################################
# ####################################################################################################################
# # ###########____________________________________________________________________###################################
x_train_, x_validation_, y_train_, y_validation_ = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

x_train_ = x_train_.to_numpy()
x_validation_ = x_validation_.to_numpy()
y_train_ = y_train_.to_numpy()
y_validation_ = y_validation_.to_numpy()

#  Convert datasets
train_dataset_ = ConvertDataset(x=torch.FloatTensor(x_train_), y=torch.FloatTensor(y_train_), train=True)
validation_dataset_ = ConvertDataset(x=torch.FloatTensor(x_validation_), y=torch.FloatTensor(y_validation_), train=True)
validation_test_dataset_ = ConvertDataset(x=torch.FloatTensor(x_validation_), train=False)
# test_dataset = ConvertDataset(x=X_test.values)

# Create DataLoaders
train_loader_ = DataLoader(dataset=train_dataset_, batch_size=batch_size_, shuffle=True)
validation_loader_ = DataLoader(dataset=validation_dataset_, batch_size=batch_size_, shuffle=True)
validation_test_loader_ = DataLoader(dataset=validation_test_dataset_, batch_size=1, shuffle=False)
# # todo: test_loader?
# # ###########____________________________________________________________________###################################
#
#
# # ###########____________________________________________________________________###################################
# Initiate model
nn_model = NN(input_size=input_size_, hidden_size=hidden_size_, num_classes=num_classes_, dropout_p=0.3).to(device)

# loss and optimizer
criterion_ = nn.BCEWithLogitsLoss()
optimizer_ = optim.Adam(nn_model.parameters(), lr=learning_rate_)

history_ = nn_model.fit(train_loader=train_loader_,
                        criterion=criterion_,
                        optimizer=optimizer_, epochs=num_epochs_,
                        validation_loader=validation_loader_, verbose=1)
# nn_model.evaluate(train_loader=train_loader_, test_loader=validation_test_loader_, y_test=y_validation_)
nn_model.evaluate(x_train=x_train_, y_train=y_train_, x_test=x_validation_, y_test=y_validation_)
#
#
# ########### report ###############
# # print(confusion_matrix(y_validation, y_pred_list))
# # print(classification_report(y_validation, y_pred_list))
# #         n_samples += labels.shape[0]
# #         n_correct += (outputs == labels).sum().item()
# #
# # ###########____________________________________________________________________###################################
