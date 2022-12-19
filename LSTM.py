import modeling1 as ex3
import nltk
from nltk import TweetTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.legacy.data import Field, LabelField, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import spacy
import string
import pickle
from tabulate import tabulate
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import re
nltk.download('stopwords')
nltk.download('wordnet')


# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvertDataset(Dataset):
    """
    Create an instances of pytorch Dataset from a DataFrame.
    """
    def __init__(self, x, lengths, y=None, train=False):
        # data loading
        self.train = train

        self.x = x
        self.lengths = lengths

        if train: self.y = y

    def __getitem__(self, index):
        if self.train:
            return self.x[index], self.lengths[index], self.y[index]
        else:
            return self.x[index]

    def __len__(self):
        return len(self.x)


class LSTM(nn.Module):
    """
    Creates an instance of LSTM model.
    Inner methods: model fit, model evaluation.
    """
    # define all the layers used in model
    def __init__(self, vocab_size_, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size_, embedding_dim=embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        """
        Feed forward the inputs data to the network's layers
        :param text: Input text
        :param text_lengths: the text original length (before padding)
        :return: predictions
        """
        # ## text = [batch size,sent_length]
        embedded = self.embedding(text)
        # ## embedded = [batch size, sent_len, emb dim]

        # ## packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # ## hidden = [batch size, num layers * num directions,hid dim]
        # ## cell = [batch size, num layers * num directions,hid dim]

        # ## concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # ## hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # ## Final activation function
        outputs = self.act(dense_outputs)

        # ------------------------------------------------- DELETE??????
        # embedded = self.embedding(text)
        # lstm_output, _ = self.lstm(embedded)
        # x = self.fc(lstm_output[:, -1, :])
        # # Final activation function
        # outputs = self.act(x)

        return outputs

    def summary(self):
        # architecture
        print(self)

    # No. of trianable parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f'The model has {count_parameters(self):,} trainable parameters')

    def fit(self, train_iterator, val_iterator, optimizer, criterion, epochs=8, verbose=0):
        """
        Fits the model to a given train set. Can be fitted with validation set as well to see accuracy and loss
        on validation throughout the training.
        :param train_iterator: of type DataLaoder. The train data (with labels)
        :param val_iterator: of type DataLaoder. Validation data (with labels)
        :param optimizer: The optimizer to use (from pytorch functions)
        :param criterion: The models criterion
        :param epochs: Number of epochs to train on
        :param verbose: Print data during training.
        :return: history. a dictionary of the models accuracy and loss of both train and validation data
                during the data fit.
        """
        history = {'accuracy': [], 'val_accuracy': [],
                   'loss': [], 'val_loss': []
                   }

        best_valid_loss = float('inf')

        # run through all epochs
        for epoch in range(1, epochs + 1):
            # train the model
            train_loss, train_acc = self._train(train_iterator, optimizer, criterion)
            print(f'epoch train acc: {train_acc}')
            # evaluate the model
            valid_loss, valid_acc = self._evaluate(val_iterator, criterion)
            print(f'epoch val acc: {valid_acc}')

            history['accuracy'].append(train_acc)
            history['loss'].append(train_loss)
            history['val_accuracy'].append(valid_acc)
            history['val_loss'].append(valid_loss)

            if verbose == 1:
                print(
                    f'Epoch {epoch}/{epochs}\n[=================] - loss: {train_loss:.5f} - accuracy: {train_acc:.4f} - val_loss: {valid_loss:.5f} - val_accuracy: {valid_acc:.4f}')

        # return history = {'accuracy': epochs_accuracy_list, 'loss': epochs_loss_list}
        return history

    def _train(self, iterator, optimizer, criterion):
        """
        Runs the inputs through the model and does backpropagation to achieve best model.
        :param iterator: Data iterator of type DataLoader
        :param optimizer: Model optimizer
        :param criterion: The criterion
        :return: Epoch's train data accuracy and loss.
        """
        self.train()  # model.train() indicates the model this is model training
        # initiate train loss and accuracy for each epoch
        epoch_loss, epoch_acc = 0, 0

        for x_batch, lengths, labels in iterator:
            # resets the gradients after every batch
            optimizer.zero_grad()

            # retrieve text and no. of words
            # text, text_lengths = batch.text

            # convert to 1D tensor
            predictions = self(x_batch, lengths)

            # compute the loss
            loss = criterion(predictions, labels.unsqueeze(1))

            # compute the binary accuracy
            acc = self._binary_acc(predictions, labels.unsqueeze(1))

            # backpropagation the loss and compute the gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        # return (train_loss, train_acc)
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def _evaluate(self, iterator, criterion):
        """
        evaluates the validation during training.
        :param iterator: data iterator
        :param criterion: models criterion.
        :return: The epochs validation accuracy and loss
        """
        # deactivating dropout layers
        self.eval()
        # initiate validation loss and accuracy for each epoch
        epoch_loss, epoch_acc = 0, 0

        # deactivates autograd
        with torch.no_grad():
            for x_batch, lengths, labels in iterator:
                # retrieve text and no. of words
                # text, text_lengths = batch.text

                # convert to 1d tensor
                # predictions = self(text, text_lengths).squeeze()
                predictions = self(x_batch, lengths)

                # compute loss and accuracy
                loss = criterion(predictions, labels.unsqueeze(1))
                acc = self._binary_acc(predictions, labels.unsqueeze(1))

                # keep track of loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        # return (val_loss, val_acc)
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

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

    def evaluate(self, train_len, test_len, x_train, x_test, y_train, y_test):
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
        train_predicted = self(x_train, train_len)
        train_acc = (train_predicted.reshape(-1).detach().numpy().round() == y_train).mean()

        test_predicted = self(x_test, test_len)
        test_predicted = test_predicted.reshape(-1).detach().numpy().round()
        # test_acc = (test_predicted == y_test).mean()
        test_acc = accuracy_score(y_test, test_predicted)
        test_prec = precision_score(y_test, test_predicted)
        test_recall = recall_score(y_test, test_predicted)
        test_auc = roc_auc_score(y_test, test_predicted)
        test_f1 = f1_score(y_test, test_predicted)

        return train_acc, test_acc, test_prec, test_recall, test_auc, test_f1

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


def lstm(train_lengths, val_lengths, train_data, valid_data, y_train, y_val, batch_size, size_of_vocab, embedding_dim,
         num_hidden_nodes,
         num_output_nodes, num_layers, directional, dropout, learning_rate, epochs, pretrained_embeddings):
    """
    This function gets data and models hyperparaetrs and create an instance of LSTM model, creates data
    iterators from the given data and fit the model.
    :param train_lengths: Train data vectors original lengths (before padding)
    :param val_lengths: Validation data vectors original lengths (before padding)
    :param train_data: Training data
    :param valid_data: Validation data
    :param y_train: train labels
    :param y_val: validation labels
    :param batch_size:  int
    :param size_of_vocab: int
    :param embedding_dim: int
    :param num_hidden_nodes: int
    :param num_output_nodes: int
    :param num_layers: int
    :param directional: bool
    :param dropout: float
    :param learning_rate: float
    :param epochs: int
    :param pretrained_embeddings:
    :return: model, history
    """
    # # Load an iterator
    train_iterator, valid_iterator = prepare_datasets(train_lengths, val_lengths, train_data, y_train, valid_data,
                                                      y_val, batch_size)

    # instantiate the model
    model = LSTM(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                 bidirectional=directional, dropout=dropout)
    # model.summary()

    # Initialize the pretrained embedding
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # print(pretrained_embeddings.shape)

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # push to cuda if available
    model, criterion = model.to(device), criterion.to(device)
    history = model.fit(train_iterator=train_iterator,
                        val_iterator=valid_iterator,
                        optimizer=optimizer,
                        criterion=criterion,
                        epochs=epochs,
                        verbose=0)

    return model, history


def prepare_datasets(train_lengths, val_lengths, x_train, y_train, x_validation, y_validation, batch_size):
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
    y_train = y_train.to_numpy()
    y_validation = y_validation.to_numpy()

    #  Convert to torch Datasets
    train_dataset = ConvertDataset(x=x_train,
                                   lengths=train_lengths,
                                   y=torch.FloatTensor(y_train),
                                   train=True)
    validation_dataset = ConvertDataset(x=x_validation,
                                        lengths=val_lengths,
                                        y=torch.FloatTensor(y_validation),
                                        train=True)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, validation_loader


def kfold_tuning(X, y, lengths, params, embeddings):
    """
    Performs a 10-Fold validation over given parameters and returns a DataFrame of the results.
    :param embeddings:
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
    tuning_val_prec = []
    tuning_val_recall = []
    tuning_val_auc = []
    tuning_val_f1 = []

    # counts number of tuning options
    options = 1
    for p in params:
        options = options * len(params[p])
    print(f'Fitting 10 folds for each of {options} candidates, totalling {options * 10} fits')

    # loop through all possible combinations
    param_values = [v for v in params.values()]
    i = 1  # index of current iteration
    for b_s, v_s, e_d, h_n, o_n, l_n, direction, dropout, lr, e in product(*param_values):
        print(
            f'{i}/{options} Tuning parameters-> | hidden_size:{h_n} | epochs:{e} | batch_size:{b_s} | learning_rate:{lr} | layers_num:{l_n} | dropout:{dropout} | bidirectional:{direction}'
        )
        i += 1

        # initiates cross-validation results
        cv_train_acc = []
        cv_val_acc = []
        cv_val_prec = []
        cv_val_recall = []
        cv_val_auc = []
        cv_val_f1 = []
        # loop through the folds
        f = 1
        for train_index, test_index in skf.split(X, y):
            # print(f)
            f = f + 1
            x_train, x_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]
            len_train, len_val = lengths[train_index], lengths[test_index]

            # initiate the neural network
            lstm_clf, history = lstm(train_lengths=len_train, val_lengths=len_val, train_data=x_train, valid_data=x_val,
                                     y_train=y_train,
                                     y_val=y_val, batch_size=b_s,
                                     epochs=e,
                                     size_of_vocab=params['VOCAB_SIZE'][0], embedding_dim=params['EMBEDDING_DIM'][0],
                                     num_hidden_nodes=h_n, num_output_nodes=1, num_layers=l_n, directional=direction,
                                     dropout=dropout, learning_rate=lr, pretrained_embeddings=embeddings)
            lstm_clf.plot_acc_loss(history)
            # evaluate on the validation
            acc = lstm_clf.evaluate(len_train, len_val, x_train, x_val, y_train.to_numpy(), y_val.to_numpy())
            # append results of the fold
            cv_train_acc.append(acc[0])
            cv_val_acc.append(acc[1])
            cv_val_prec.append(acc[2])
            cv_val_recall.append(acc[3])
            cv_val_auc.append(acc[4])
            cv_val_f1.append(acc[5])
        print(
            f'train_acc: {np.mean(cv_train_acc):.3f}, val_acc:{np.mean(cv_val_acc):.3f}, val_prec:{np.mean(cv_val_prec):.3f}, val_recall:{np.mean(cv_val_recall):.3f}, val_auc:{np.mean(cv_val_auc):.3f}, val_f1:{np.mean(cv_val_f1):.3f}')
        iter_params = {'hidden_size': h_n, 'hidden_size': h_n, 'epochs': e,
                       'batch_size': b_s, 'learning_rate': lr, 'layers_num': l_n, 'dropout': dropout,
                       'bidirectional': direction}
        # append results of the iteration
        tuning_params.append(iter_params)
        tuning_val_acc.append(np.mean(cv_val_acc))
        tuning_train_acc.append(np.mean(cv_train_acc))

    cv_results = {'params': tuning_params, 'mean_test_score': tuning_val_acc, 'mean_train_score': tuning_train_acc}
    return cv_results


def lstm_tuning(train_lengths, x_train, y_train, params_grid, embedding_vector):
    """
    Preforms parameters tuning on the ann
    :param x_train: train data frame
    :param y_train: train labels data frame
    :param params_grid: tuning parameters (dictionary)
    :return: Best model
    """
    results = kfold_tuning(X=x_train, y=y_train, lengths=train_lengths, params=params_grid, embeddings=embedding_vector)
    # convert dictionary to DataFrame
    results = pd.DataFrame(results).sort_values('mean_test_score', ascending=False)
    # print table
    headers = ['Parameters', 'Validation score', 'Train score']
    print(tabulate(results.head(10), headers=headers, tablefmt='grid'))

    return results.head(1)


################################### Prepare data and make Embeddings for LSTM ########################################

def read_data_for_embedding():
    """
    The function reads the data and turns it into a dataframe of - label tweet only.
    """
    #### train data ####
    train_data = ex3.read_data('trump_train.tsv')  # read train data
    train_embed_data = train_data.drop(['id', 'user', 'time', 'device'], axis=1)
    x_train = train_embed_data.drop('label', axis=1)
    y_train = train_embed_data['label']

    ###### test data ####
    test_data = ex3.read_data('trump_test.tsv', True)  # read test data
    x_test = test_data.drop(['user', 'time'], axis=1)

    return x_train, y_train, x_test


def preprocess_tweets_for_embedding(x_emb_train):
    """
    This function get the train data, clean and normalized the tweets before make embedding:
    Lower case
    Removing urls
    Removing hashtags
    Removing punctuation
    Removing stop words
    Removing stock-market symbols
    Lemmatizating

    :param x_emb_train: DataFrame of the train data , contain the tweets
    :return: Dataframe with clean tweets.
    """
    clean_tweets = []
    for tweet in x_emb_train['tweet']:
        tweet_lower_case = tweet.lower()  # tweet to lowercase
        remove_stock_market_symbols = re.sub(r'\$\w*', '', tweet_lower_case)  # remove stock-market
        remove_hashtags = re.sub(r'#', '', remove_stock_market_symbols)  # remove hashtag
        remove_url = re.sub(r'https?:\/\/.*[\r\n]*', 'url', remove_hashtags)  # remove urls
        remove_sw = remove_stop_words(remove_url)  # remove stopwords
        remove_punc = remove_punctuation(remove_sw)  # remove punctuation
        tweet_tokenize = TweetTokenizer(strip_handles=True, reduce_len=True)
        tweet_tokens_list = tweet_tokenize.tokenize(remove_punc)
        tweett_lemmatize = lemmatizing(tweet_tokens_list)
        clean_tweet = tweett_lemmatize

        clean_tweets.append(clean_tweet)

    return clean_tweets


def remove_stop_words(tweet):
    """
    This function remove stopwords from tweet
    :param tweet: str for clean
    :return: clean tweet
    """
    STOPWORDS = set(stopwords.words("english"))
    tweet = " ".join(word for word in tweet.split() if word not in STOPWORDS)
    return tweet


def remove_punctuation(tweet):
    """
    This function remove punctuation from tweet
    :param tweet: str for clean
    :return: clean tweet
    """
    tweet = "".join(word for word in tweet if word not in set(string.punctuation))
    return tweet


def lemmatizing(tokenized_text):
    """
    this function lemmatizing each word token in the list input.
    :param tokenized_text: list of tokens
    :return: lemmatize tokens list
    """
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text


########################### create the embeddings #################################


def make_embedding(clean_data):
    """
    This function make embedding by using glove.twitter.27B.100d
    :return: embedding matrix of our tweets
    """

    TEXT = Field(sequential=True, batch_first=True, include_lengths=True, fix_length=24)
    data = list(map(TEXT.preprocess, clean_data))
    data = TEXT.pad(data)
    TEXT.build_vocab(data[0], vectors='glove.twitter.27B.100d')
    vocab_size = len(TEXT.vocab)
    data_for_embedding, length = TEXT.numericalize(data)
    # print(data_for_embedding , length)

    embedding_vectors = TEXT.vocab.vectors

    return embedding_vectors, vocab_size, data_for_embedding, length


####### read and make data to preprocessing #######

# X_embedding_train, Y_embedding_train, X_embedding_test = read_data_for_embedding()
#
# X_emb_clean_train = preprocess_tweets_for_embedding(X_embedding_train)
# ############### make embedding #################
# embedding_vectors, vocab_size, x_data, data_lengths = make_embedding(X_emb_clean_train)
#
# params = {'BATCH_SIZE': [64],
#           'VOCAB_SIZE': [vocab_size],
#           'EMBEDDING_DIM': [100],
#           'HIDDEN_NODES': [32],
#           'OUTPUT_NODES': [1],
#           'lAYERS_NUM': [2],
#           'BIDIRECTIONAL': [True],
#           'DROPOUT': [0.1],
#           'LR': [0.001],
#           'EPOCHS': [2]
#           }
#
# best_params = {'BATCH_SIZE': [16],
#                'VOCAB_SIZE': [vocab_size],
#                'EMBEDDING_DIM': [100],
#                'HIDDEN_NODES': [64],
#                'OUTPUT_NODES': [1],
#                'lAYERS_NUM': [2],
#                'BIDIRECTIONAL': [True],
#                'DROPOUT': [0.2],
#                'LR': [0.01],
#                'EPOCHS': [10]
#                }

# lstm_tuning(train_lengths=data_lengths, x_train=x_data, y_train=Y_embedding_train, params_grid=best_params,
#             embedding_vector=embedding_vectors)
