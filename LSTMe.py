from nltk import TweetTokenizer
import torch.nn as nn
import ex3_307887984_307830901 as ex3
import torch
import torchtext
from torchtext.legacy.data import Field, LabelField, BucketIterator , TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import spacy
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
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import re
import nltk
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import Tokenizer

from torch.nn.utils.rnn import pad_sequence

from tensorflow.python import tf2

# nltk.download('stopwords')



# training_data = ex3.read_data('trump_train.tsv')
# training_data = training_data[['tweet', 'label']]
#
# TEXT = Field(tokenize='moses', batch_first=True, include_lengths=True)
# # LABEL = LabelField(dtype=torch.float, batch_first=True)
#
# fields = [('tweet', TEXT), ('label', LABEL)]
#
# train_data_, valid_data_ = training_data.split(split_ratio=0.7, random_state=1)

# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set parameters
BATCH_SIZE = 64
# SIZE_OF_VOCAB = len(TEXT.vocab)
EMBEDDING_DIM = 100
NUM_HIDDEN_NODES = 32
NUM_OUTPUT_NODES = 1
NUM_LAYERS = 2
BIDIRECTION = True
DROPOUT = 0.2


class LSTM(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation function
        self.act = nn.Sigmoid()

    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs

    def summary(self):
        # architecture
        print(self)

        # No. of trianable parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'The model has {count_parameters(self):,} trainable parameters')

    def fit(self, train_iterator, val_iterator, optimizer, criterion, epochs=8, verbose=0):
        history = {'accuracy': [], 'val_accuracy': [],
                   'loss': [], 'val_loss': []
                   }

        best_valid_loss = float('inf')

        # run through all epochs
        for epoch in range(1, epochs + 1):
            # train the model
            train_loss, train_acc = self._train(train_iterator, optimizer, criterion)
            # evaluate the model
            valid_loss, valid_acc = self._evaluate(val_iterator, criterion)

            history['accuracy'].append(train_acc)
            history['loss'].append(train_loss)
            history['val_accuracy'].append(valid_acc)
            history['val_loss'].append(valid_loss)

            # save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.state_dict(), 'saved_weights.pt')

            if verbose == 1:
                print(
                    f'Epoch {epoch}/{epochs}\n[=================] - loss: {train_loss:.5f} - accuracy: {train_acc:.4f} - val_loss: {valid_loss:.5f} - val_accuracy: {valid_acc:.4f}')

        # return history = {'accuracy': epochs_accuracy_list, 'loss': epochs_loss_list}
        return history

    def _train(self, iterator, optimizer, criterion):
        self.train()  # model.train() indicates the model this is model training
        # initiate train loss and accuracy for each epoch
        epoch_loss, epoch_acc = 0, 0

        for batch in iterator:
            # resets the gradients after every batch
            optimizer.zero_grad()

            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1D tensor
            predictions = self(text, text_lengths).squeeze()

            # compute the loss
            loss = criterion(predictions, batch.label)

            # compute the binary accuracy
            acc = self._binary_accuracy(predictions, batch.label)

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
        # deactivating dropout layers
        self.eval()
        # initiate validation loss and accuracy for each epoch
        epoch_loss, epoch_acc = 0, 0

        # deactivates autograd
        with torch.no_grad():
            for batch in iterator:
                # retrieve text and no. of words
                text, text_lengths = batch.text

                # convert to 1d tensor
                predictions = self(text, text_lengths).squeeze()

                # compute loss and accuracy
                loss = criterion(predictions, batch.label)
                acc = self._binary_accuracy(predictions, batch.label)

                # keep track of loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        # return (val_loss, val_acc)
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def _binary_accuracy(self, preds, y):
        # round predictions to the closest integer
        rounded_preds = torch.round(preds)

        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
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




def lstm(train_data, valid_data, batch_size, size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes,
         num_layers, directional, dropout, learning_rate, epochs ,  pretrained_embeddings):
    # Load an iterator
    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data),
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)

    # instantiate the model
    model = LSTM(size_of_vocab, embedding_dim, num_hidden_nodes, num_output_nodes, num_layers,
                 bidirectional=directional, dropout=dropout)
    # model.summary()
    #
    # Initialize the pretrained embedding
    # pretrained_embeddings = TEXT.vocab.vectors # todo
    model.embedding.weight.data.copy_(pretrained_embeddings)

    # print(pretrained_embeddings.shape)

    # define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # push to cuda if available
    model, criterion = model.to(device), criterion.to(device)
    history = model.fit(train_iterator=train_iterator, val_iterator=valid_iterator, optimizer=optimizer,
                        criterion=criterion,
                        epochs=epochs, verbose=1)

    return model, history


def kfold_tuning(X, y, params, emb):
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


def lstm_tuning(x_train, y_train, params_grid, embedding_vector):
    """
    Preforms parameters tuning on the ann
    :param x_train: train data frame
    :param y_train: train labels data frame
    :param params_grid: tuning parameters (dictionary)
    :return: Best model
    """
    results = kfold_tuning(X=x_train, y=y_train, params=params_grid, emb=embedding_vector)
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

    :param x_emb_train: DataFrame of the train data , contain the tweets
    :return: Dataframe with clean tweets.
    """
    clean_tweets = []
    for tweet in x_emb_train['tweet']:
        tweet_lower_case = tweet.lower()  # tweet to lowercase
        remove_stock_market_symbols = re.sub(r'\$\w*', '', tweet_lower_case)  # remove stock-market
        remove_hashtags = re.sub(r'#', '', remove_stock_market_symbols)  # remove hashtag
        remove_url = re.sub(r'https?:\/\/.*[\r\n]*', '', remove_hashtags)  # remove urls
        remove_sw = remove_stop_words(remove_url)  # remove stopwords
        remove_punc = remove_punctuation(remove_sw)  # remove punctuation
        tweet_tokenize = TweetTokenizer(strip_handles=True, reduce_len=True)
        tweet_tokens_list = tweet_tokenize.tokenize(remove_punc)
        clean_tweet = tweet_tokens_list

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


########################### create the embeddings #################################


def make_embedding(clean_data):
    """
    This function make embedding by using glove.twitter.27B.100d
    :return: embedding matrix of our tweets
    """


    TEXT = Field(sequential=True, batch_first=True, include_lengths=True, fix_length=24)
    data = list(map(TEXT.preprocess, clean_data))
    data = TEXT.pad(data)
    TEXT.build_vocab(data, vectors='glove.twitter.27B.100d')
    vocab_size = len(TEXT.vocab)
    data_for_embedding = TEXT.numericalize(data)
    embedding_vectors = TEXT.vocab.vectors

    return data_for_embedding , embedding_vectors , vocab_size




####### read and make data to preprocessing #######
# TODO make everything for the test too!

X_embedding_train, Y_embedding_train, X_embedding_test = read_data_for_embedding()

X_emb_clean_train = preprocess_tweets_for_embedding(X_embedding_train)
############### make embedding #################

data_for_lstm , embedding_vectors , vocab_size= make_embedding(X_emb_clean_train)

params = {'BATCH_SIZE': [32],
          'VOCAB_SIZE': [vocab_size],
          'EMBEDDING_DIM': [100],
          'HIDDEN_NODES': [32],
          'OUTPUT_NODES': [1],
          'lAYERS_NUM': [1],
          'BIDIRECTIONAL': [False],
          'DROPOUT': [0.2],
          'LR': [0.01],
          'EPOCHS': [8]
          }


lstm_tuning(data_for_lstm, Y_embedding_train, params, embedding_vectors)

