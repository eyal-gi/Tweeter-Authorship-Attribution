import csv
import nltk
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import TweetTokenizer
from datetime import datetime
from collections import Counter
from dython import nominal
from sklearn.preprocessing import MinMaxScaler
import copy
import math


# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


########################################################################################################################
# --------------------------------------- Data Understanding & Pre Processing ----------------------------------------#
########################################################################################################################


def load_best_model():
    pass


def train_best_model():
    pass


def predict(m, fn):
    pass


def read_data(data_name, test=False):
    """
    Reads tsv file and returns a pandas DataFrame.
    :param data_name: The name of the file.
    :param test: boolean. True is reading is done to test file.
    :return: pandas DataFrame
    """
    if not test:
        names = ['id', 'user', 'tweet', 'time', 'device']

    else:
        names = ['user', 'tweet', 'time']

    data = pd.read_csv(data_name, sep='\t', header=None, names=names,
                       quoting=csv.QUOTE_NONE)

    #   ##make labels : where the device is android - 0 (Trump), else - 1 (staff)
    if not test:
        label_list = []
        for device in data['device']:
            if device == 'android':
                label_list.append(0)
                # label_list.append('Trump')
            else:
                label_list.append(1)
                # label_list.append('Not Trump')
        data.insert(0, 'label', label_list)
    return data


def preliminary_feature_extraction(df):
    """
    Preliminary creation of features out of the data.
    The extracted features are: negative score of each tweet (float), Use of quotes in the tweet (boolean),
    Publish hour(time), Number of hashtags in the tweet (int), URL in the tweet (boolean),
    Events times (boolean),	2 or more Exclamation mark (int), Tweet length (int),
    Normalized Number of capital lettered words (float), number of words beginning with capital letter (float),
    Tagging other users [pattern of ‘@:xxxxx’] (int), Tagging himself [pattern of ‘@realDonaldTrump’] (boolean),
    Percentage of use of a particular tag in a sentence - pos tag (float).

    :param df: DataFrame
    :return: Updated DataFrame with new features
    """
    pfe_df = copy.deepcopy(df)
    # #### Patterns Features: ####
    patterns_features(pfe_df)
    # #### Capital letters Features: ####
    count_capital_words(pfe_df)
    # #### Tags himself (#realDonaldTrump) in tweet
    find_pattern(pfe_df, None, 'tag_realDonaldTrump')
    # #### Negative score for each tweet (knows that trump tweets are more negative than his staff): ####
    negative_score_for_tweet_feature(pfe_df)
    # #### length of tweet (the length of trump tweets are on average 7 words longer than the rest): ####
    tweet_length_feature(pfe_df)
    ## Checking the time that the tweets were published
    publish_time_feature(pfe_df)
    # #### Pos Tag feature: ####
    pos_tag_feature(pfe_df)

    return pfe_df


def patterns_features(data):
    """
    Creates all pattern-like features
    :param data: DataFrame
    :return: Updated DataFrame with pattern-like features
    """
    # ## count number of users tags in a tweet:
    find_pattern(data, pattern=r'@\S+', col_name='tags_count')
    # ## count number of hashtags in a tweet:
    find_pattern(data, pattern=r'#\S+', col_name='hashtags_count')
    # ## mark tweets with quotes:
    find_pattern(data, pattern=r'".*?"', col_name='quotes')
    # ## mark tweets with url attached:
    find_pattern(data, pattern=r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                 col_name='url')
    # ## mark tweets with hour:
    find_pattern(data, pattern=r'([01]?[0-9]|2[0-3]):[0-5][0-9]', col_name='written_time')
    # ## count number of over 2 exclamation marks in tweets:
    find_pattern(data, pattern=r'!!+', col_name='ex_mark')

    return data


def find_pattern(df, pattern, col_name):
    """
    Creates a pattern-like feature
    :param df: DataFrame
    :param pattern: The pattern to create the feature by
    :param col_name: Name of the feature
    :return: Updated DataFrame with the desired feature
    """
    counts = []
    tweets = df['tweet']
    for t in tweets:  # go over all the tweets
        if pattern is not None:  # pattern was given
            match = re.findall(pattern, t)  # finds all strings that matches the given pattern

        if col_name == 'tags_count' or col_name == 'hashtags_count' or col_name == 'ex_mark':
            counts.append(len(match))  # append the number of occurrences
        elif col_name == 'quotes' or col_name == 'url':
            counts.append(np.sign(len(match)))  # append 0 if lean=0 or 1 if len>1
        elif col_name == 'written_time':
            match = re.search(pattern, t)  # boolean -> True is pattern found in string, False if not
            counts.append(int(match is not None))  # append the boolean as int (True:1, False:0)
        elif col_name == 'tag_realDonaldTrump':
            tt = TweetTokenizer()  # tweeter tokenizer
            tokens = tt.tokenize(t)  # tweet tokens
            flag = False  # flag represents if trump's tag been found
            for t in tokens:
                if t == '@realDonaldTrump':
                    flag = True
                    break
            counts.append(int(flag))  # append 0 if tag not found, 1 if tag found

    df[col_name] = counts  # add the feature column to the data frame

    return df


def count_capital_words(df):
    """
    Creates a feature of capital lettered words and words starting with capital letter.
    :param df: DataFrame
    :return: Updated DataFrame with 2 more features
    """
    col_name1 = 'full_cap_words_count'
    col_name2 = 'cap_words_count'
    tt = TweetTokenizer()
    counts1 = []
    counts2 = []
    tweets = df['tweet']
    for t in tweets:
        tokens = tt.tokenize(t)
        counter1 = 0
        counter2 = 0
        for t in tokens:
            counter1 += int(t.isupper())  # add 1 if token is all uppercase , 0 if not
            counter2 += int(t[0].isupper())  # add 1 if token start with uppercase letter, 0 if not
        counts1.append(counter1 / len(tokens))  # normalize by tweet token's size
        counts2.append(counter2 / len(tokens))  # normalize by tweet token's size

    # add the features to the data frame:
    df[col_name1] = counts1
    df[col_name2] = counts2

    return df


def negative_score_for_tweet_feature(df):
    """
    This function calculate the negative score of the tweet and return the df with the new feature.
    :param df: df with all the features until now
    :return: df: the df with the new feature

    By reading the article we know that trump tweets are more negative than his staff, so we want to add this feature
    and check if it can help us.
    """
    sia = SentimentIntensityAnalyzer()
    neg_score_list = []
    for tweet in df['tweet']:  # run over tweets
        tweet_score = sia.polarity_scores(tweet)
        neg_score = tweet_score['neg']  # take the negative score
        neg_score_list.append(neg_score)
    df['negative_score'] = neg_score_list  # add the negative_score feature to our df

    return df


def tweet_length_feature(df):
    """
    This function calculate the length of each tweet and return the df with the new feature.
    :param df: df with all the features until now
    :return: df: the df with the new feature
    """
    # This function use nltk tweet tokenizer and not the regular tokenizer
    length_list = []
    tt = TweetTokenizer()
    for tweet in df['tweet']:
        tweet_tokens = tt.tokenize(tweet)
        tweet_length = len(tweet_tokens)  # calculate the length of the tweet by count the number of tokens
        length_list.append(tweet_length)
    df['tweet_length'] = length_list

    return df


def publish_time_feature(df):
    """
    This function check at what time(hour) the tweet was published and return the df with the new feature.
    :param df: df with all the features until now
    :return: df: the df with the new feature

    The function use datatime to get the hour of publish and use it as feature.
    """
    time_list = []
    df['time'] = pd.to_datetime(df['time'])  # convert to datetime
    for date_time in df['time']:
        date_time = date_time.hour  # take only the hour of publish for the timestamp
        time_list.append(date_time)
    df['hr_publish_time'] = time_list

    return df


def pos_tag_feature(df):
    """
    This function take each token in the every tweet and give him tag and return the df with the new feature.
    :param df: df with all the features until now
    :return: df: the df with the new feature

    This function use pos_tag function and then calculates the percentage of use of a particular tag in a sentence
    (from specific list of tags)
    """
    pos_list = ['NN', 'DT', 'IN', 'JJ', 'NNS', 'VBZ', 'VBD']  # make list of relevant tags after try all the tags.
    tags_dict = {'NN': [], 'DT': [], 'IN': [], 'JJ': [], 'NNS': [], 'VBZ': [], 'VBD': []}  # help dictionary
    tweet_counter = 0
    tt = TweetTokenizer()
    for tweet in df['tweet']:
        tweet_counter += 1
        tweet_tokens = tt.tokenize(tweet)
        tweet_length = len(tweet_tokens)
        count = Counter([j for i, j in nltk.pos_tag(tweet_tokens)])
        for tag in count.keys():
            if tag in pos_list:
                pos_percent = count[tag] / tweet_length
                tags_dict[tag].append(pos_percent)
        for tag in tags_dict.keys():
            if len(tags_dict[tag]) != tweet_counter:
                tags_dict[tag].append(0.0)
    for tag in tags_dict.keys():
        df[tag] = tags_dict[tag]

    return df



def feature_understanding(df):
    """
    Visualise the features to better understand the data.
    :param df: DataFrame
    """
    pd.options.display.max_columns = 10
    pd.options.display.width = 1000
    features = ['tags_count', 'hashtags_count', 'quotes', 'url', 'written_time', 'ex_mark',
                'tag_realDonaldTrump',
                'full_cap_words_count', 'cap_words_count', 'negative_score', 'tweet_length', 'hr_publish_time', 'NN',
                'DT', 'IN', 'JJ', 'NNS', 'VBZ', 'VBD']
    for f in features:
        features_plots(df, f)


def features_plots(df, col_name):
    """
    Plots a feature. Based on the feature name, a suitable plot will be presented
    :param df: DataFrame
    :param col_name: Name of feature
    """
    m_df = df[['label', col_name]]
    sums = m_df.groupby('label')[col_name].sum()
    # print(sums)

    #### BOX-PLOTS & BAR-PLOTS #####
    # ---------------------#
    if col_name == 'tags_count' or col_name == 'hashtags_count' or col_name == 'ex_mark' or col_name == 'negative_score' \
            or col_name == 'tweet_length' or col_name == 'hr_publish_time' or col_name == 'NN' or col_name == 'DT' \
            or col_name == 'IN' or col_name == 'JJ' or col_name == 'NNS' or col_name == 'VBZ' or col_name == 'VBD':
        sns.boxplot(x='label', y=col_name, data=m_df, palette='Set2', showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
        plt.title(col_name)
    elif col_name == 'quotes' or col_name == 'url' or col_name == 'written_time' or col_name == 'tag_realDonaldTrump':
        sns.catplot(x=col_name, hue="label", kind="count", data=m_df, palette='Set2')
        plt.title(col_name)
        label_0_sum = sum(m_df.label == 0)
        label_1_sum = sum(m_df.label == 1)
        sums = m_df.groupby('label')[col_name].sum()

        # print('trump:', label_0_sum, '| not trump:', label_1_sum, '| trump', col_name, ':', sums[0], '| not trump',
        #       col_name, ':', sums[1])
        normalized_col_name = col_name + '_count_normalized'
        plot_df = pd.DataFrame({'label': ['Trump', 'Trump', 'Not Trump', 'Not Trump'],
                                col_name: [0, 1, 0, 1],
                                normalized_col_name: [1 - sums[0] / label_0_sum,
                                                      sums[0] / label_0_sum,
                                                      1 - sums[1] / label_1_sum,
                                                      sums[1] / label_1_sum]})
        sns.catplot(data=plot_df, kind='bar', x=col_name, y=normalized_col_name, hue='label', palette='Set2')
        plt.title(col_name + ' Normalized')

    elif col_name == 'full_cap_words_count' or col_name == 'cap_words_count':
        sns.boxplot(x='label', y=col_name, data=df, palette='Set2', showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
        plt.title(col_name)

    plt.show()

    #### HISTOGRAMS #####
    # ---------------------#
    if col_name == 'hr_publish_time':
        x = [df.loc[df.label == 0, 'hr_publish_time'],
             df.loc[df.label == 1, 'hr_publish_time']]
        plt.hist(x=x, bins=24, density=True, histtype='step', label=['Trump', 'Not Trump'])
        plt.xticks(ticks=range(24),
                   labels=['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00',
                           '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00',
                           '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'])
        plt.title('Tweet published hour')
        plt.xlabel('Hour')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


def normalize_features(df, column_name_list: list):
    """
    This function rum over specific columns by the column_name_list and normalizes the column by the max value in it.
    :param df: DataFrame
    :param column_name_list: list of columns names to normalized
    :return: df with normalized columns
    """
    for col in column_name_list:
        scaler = MinMaxScaler()
        scaler.fit(df[[col]])
        df[col] = scaler.transform(df[[col]]).round(3)
    return df


def make_categorized_from_hr_publish(df):
    """
    This function make a categories from the hr_publish_time to night-morning / noon / evening
    :param df: DataFrame
    :return: df with hr_publish_time that is category
    """
    temp_df = copy.deepcopy(df)
    for i in range(len(df)):
        if temp_df.at[i, 'hr_publish_time'] == 0:
            temp_df.at[i, 'hr_publish_time'] = 24
        if temp_df.at[i, 'hr_publish_time'] == 1:
            temp_df.at[i, 'hr_publish_time'] = 25
        if temp_df.at[i, 'hr_publish_time'] == 2:
            temp_df.at[i, 'hr_publish_time'] = 26

    df = temp_df
    df['hr_publish_time'] = pd.cut(df['hr_publish_time'], bins=[3, 10, 16, 26],
                                   labels=['morning', 'noon', 'evening-night'])

    return df


def set_type_for_features(df):
    """
    This fucntion get df and convert relevant features to a new type
    :param df: DataFrame
    :return: df with the correct type of features

    """
    df = df.astype({'label': bool, 'quotes': bool, 'url': bool, 'written_time': bool, 'tag_realDonaldTrump': bool})

    return df


def feature_correlation(df):
    """
    Correlation of continuous variables versus continuous variables was examined by the Pearson correlation.
    Correlation of categorical variables versus categorical variables examined by the Cramer's correlation (Cramer's V)
    and correlation between continuous variables to categorical examined by match ratio (ratio correlation.)

    :param df: DataFrame
    :return: Heatmap of correlations
    """
    temp_df = df
    temp_df = temp_df.drop(['id', 'user', 'tweet', 'time', 'device'],
                           axis=1)  # drop not relevant featurs before make a heatmap of correlations
    nominal.associations(temp_df, nominal_columns='all')


def feature_selection(df, test_flag = False):
    """
    This function get df, drop the features that we understand that are not relevant or good enough for us and return
    final df for train and test.

    :param df: DataFrame
    :return: The final Dataframes for train and test
    """
    # Prepare data
    final_df = copy.deepcopy(df)
    normalize_features(final_df, ['hashtags_count' , 'tweet_length' , 'tags_count']) # normalized the features to be between [0,1].
    final_df = make_categorized_from_hr_publish(final_df) #make categories from hr_publish_time feature
    final_df = pd.get_dummies(final_df, columns =['hr_publish_time'], drop_first=True) # get dummies for categorical feature
    # Features selection
    if not test_flag:
        final_df = final_df.drop('id',axis=1)
        final_df = final_df.drop('user',axis=1)
        final_df = final_df.drop('tweet',axis=1)
        final_df = final_df.drop('time',axis=1)
        final_df = final_df.drop('device',axis=1)
        final_df = final_df.drop('written_time',axis=1)
        final_df = final_df.drop('ex_mark',axis=1)
        # final_df.to_csv('train_df.csv')
    else:
        final_df = final_df.drop('user', axis=1)
        final_df = final_df.drop('tweet', axis=1)
        final_df = final_df.drop('time', axis=1)
        final_df = final_df.drop('written_time',axis=1)
        final_df = final_df.drop('ex_mark',axis=1)
        # final_df.to_csv('test_df.csv')

    return final_df






def pre_process_main():
    ########### Train data - read and make features ###########
    train_data = read_data('trump_train.tsv')  # read train data
    train_data_fe = preliminary_feature_extraction(train_data)  # feature extraction for train data
    ########### Test data - ead and make features ###########
    test_data = read_data('trump_test.tsv' , True) # read test data
    test_data_fe = preliminary_feature_extraction(test_data)  # feature extraction for train data

    ########### Data Understanding - plots and correlation ###############
    # feature_understanding(train_data_fe)
    # normalize_features(train_data_fe, ['hashtags_count' , 'tweet_length' , 'tags_count']) # normalized the features to be between [0,1].
    # train_data_fe = make_categorized_from_hr_publish(train_data_fe)
    # train_data_fe = pd.get_dummies(train_data_fe, columns =['hr_publish_time'], drop_first=True)
    ######################################## For correlation ##########################################################
    # train_data_fe=set_type_for_features(train_data_fe) # make the type boolean for the correlation heatmap
    # feature_correlation(train_data_fe) #make a heatmap correlation
    ###################################################################################################################

    ################# Feature Selection And Make Final Train and Test Df #################
    final_train_df = feature_selection(train_data_fe)
    final_test_df = feature_selection(test_data_fe,True)

    return final_train_df , final_test_df

########################################################################################################################
# ----------------------------------------------------- Models -------------------------------------------------------#
########################################################################################################################

######### split for features and labels ############
def read_and_split_data():
    """
    This function read the data after features selection and split it to X_train , Y_train and for X_test
    """
    train_df = pd.read_csv('train_df.csv')
    X_test = pd.read_csv('test_df.csv')
    X_train = train_df.drop('label', axis=1)
    Y_train = train_df[['label']]

    return X_train ,Y_train , X_test





if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    # pre_process_main()
    read_and_split_data()
