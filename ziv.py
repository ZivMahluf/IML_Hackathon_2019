import twitter as twitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
import seaborn as sns

from visualize_data import ZivsVisualizer as zv
from process_data import ZivsProcessor as zp
from itertools import product
from sklearn import tree
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

import os
from four_c import delete_stop_words_create_stems
from vitaly import process_sentence
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from collections import defaultdict

Trump = {'great', 'Border', 'President', 'people', 'Democrats', 'Trump',
         'Country', 'US', 'Wall', 'Fake', 'Great', 'Mueller', 'China',
         'States', 'United'}

Joe_Biden = {'Biden', 'VP', 'President', 'Joe', 'country', 'Obama',
             'America', 'middle', 'Vice'}

Conan_Obrian = {'show', 'know', 'Trump', 'say', 'dont'}

Ellen_Degeneres = {'Happy', 'birthday', 'love', 'show', '#GameofGames',
                   '#ThanksSponsor', 'gonna', 'full', 'hope'}

Kim_Kardeshian = {'PST', '12PM', '#KKWBEAUTY', 'Shop', '@kkwbeauty',
                  '@KimKardashian', 'available', 'Crme', 'Classic', 'Lip',
                  'Happy', 'Pop-Up', '#KUWTK', 'love', 'Lipstick', 'KKW',
                  'Collection'}

Lebrom_James = {'@KingJames', 'u', '#StriveForGreatness', 'Conqrats',
                'game', 'S/O', '#Striveforgreatne', 'Man', 'homie', 'Love',
                'guys'}

Lady_GaGa = {'Gaga', 'Lady', 'love', 'u', '@ladygaga', 'Thank', 'Tony',
             'world', '#JOANNE', 'beautiful', 'music'}

Cristiano_Ronaldo = {'@Cristiano', 'game', 'win', 'Thank', 'eat', 'de',
                     'team', 'asked:', 'Check', 'support', 'fans', 'CR7'}

Jimmy_Kimmel = {'@RealDonaldTrump', '@IamGuillermo', 'great', '@jimmykimmel',
                'show', '@TheCousinSal', 'ever'}

Schwarzeneger = {'@Schwarzenegger', 'Arnold', 'great', 'fantastic', 'love',
                 'proud', '@ArnoldSports', 'gerrymandering'}

Sets_mapping = [Trump, Joe_Biden, Conan_Obrian, Ellen_Degeneres, Kim_Kardeshian, Lebrom_James,
                Lady_GaGa, Cristiano_Ronaldo, Jimmy_Kimmel, Schwarzeneger]


TYPE_OF_DATA = 'train'
if __name__ == '__main__':
    # data = pd.DataFrame()
    # if TYPE_OF_DATA == 'train':
    #     for filename in os.listdir('tweets_data'):
    #         if filename.endswith('.csv'):
    #             file_data = pd.read_csv('tweets_data/' + filename)
    #             file_data['user'].fillna(
    #                 file_data['user'].mode(dropna=True))  # use only for traindata
    #             file_data['tweet'].dropna(inplace=True)
    #             data = data.append(file_data, ignore_index=True)
    # else:
    #     pass  # handle test (its given as a list of tweets/strings
    #
    # data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))
    # data['tweet'] = data['tweet'].apply(lambda w: w[2:len(w) - 2])  # remove [' and ']
    #
    # vitaly = process_sentence(data['tweet'])
    # data = pd.concat([data, vitaly], axis=1)
    # labels = pd.Series(data['user'])
    # data = pd.get_dummies(data, columns=['user'])  # binary features for the user
    # # data = data.drop(columns=['words', 'tweet'])
    # # data['common_words'] = delete_stop_words_create_stems(data['words'])

    # TODO: replaced by manual sets
    # nltk.download('stopwords', quiet=True)
    # stop_words = set(stopwords.words('english'))
    # stop_words = stop_words.union({'&amp;', 'new', 'one', 'time'})
    # user_common_word_set = dict()
    # for i in range(10):
    #     count_vect = defaultdict(int)
    #     user_indices = data[data['user_' + str(i)] == 1].index
    #     words = data.iloc[user_indices]['words']
    #     for word_list in words:
    #         for x in word_list:
    #             if x.lower() not in stop_words:  # remove stopwords from 'words' column
    #                 count_vect[x] += 1
    #     v = sorted(count_vect.items(), key=lambda kv: kv[1],
    #                reverse=True)  # sort ascending
    #     user_common_word_set[i] = v[0:100]
    # # TODO: save user common word set to an external file, use as feature afterwards
    # for j in range(10):
    #     # save
    #     f = open('sets/set_user_' + str(j), 'w')
    #     f.write(str(user_common_word_set[j]))
    #     f.close()

    # # add column to model based on set of words
    # set_features = pd.DataFrame(columns=['set_' + str(k) + '_score' for k in range(10)])
    # print("shape of data " + str(data.shape))
    # for word_list_index, word_list in data['words'].iteritems():
    #     print(word_list_index)
    #     score = defaultdict(float)
    #     for word in word_list:
    #         for g in range(10):
    #             if word in Sets_mapping[g]:
    #                 score[g] += 1
    #     set_features = set_features.append(score, ignore_index=True)
    #
    # data = pd.concat([data, set_features], axis=1)
    # data.to_csv('sets/data_after_set_features.csv')
    # print()

    # # replace the empty values with 0
    # data = pd.read_csv('sets/data_after_set_features.csv')
    # data = data.fillna(0)
    # for w in data.columns:
    #     if w.find('set') != -1:
    #         data[w] = data[w].fillna(0).astype(int)
    # data.to_csv('sets/new_data_thursday_night.csv')

    # delete the user columns and replace them by labels
    data = pd.read_csv('sets/new_data_thursday_night.csv')
    labels = np.zeros(data.shape[0], dtype=int)
    for k in range(10):
        w = 'user_' + str(k)
        labels += (k * data[w])
        data = data.drop(columns=[w])

    unnamed_cols = [col for col in data.columns if col.find('Unnamed') != -1]
    data = data.drop(columns=unnamed_cols)
    data['user'] = labels
    data.to_csv('sets/data_with_user.csv')

    print()
