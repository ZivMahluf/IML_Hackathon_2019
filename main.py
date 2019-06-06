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

TYPE_OF_DATA = 'train'
if __name__ == '__main__':
    data = pd.DataFrame()
    if TYPE_OF_DATA == 'train':
        for filename in os.listdir('tweets_data'):
            if filename.endswith('.csv'):
                file_data = pd.read_csv('tweets_data/' + filename)
                file_data['user'].fillna(
                    file_data['user'].mode(dropna=True))  # use only for traindata
                file_data['tweet'].dropna(inplace=True)
                data = data.append(file_data)
    else:
        pass  # handle test (its given as a list of tweets/strings

    data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))

    # minimal preprocessing on the strings
    data['tweet'] = data['tweet'].apply(lambda w: w[2:len(w) - 2])  # remove [' and ']

    vitaly = process_sentence(data['tweet'])
    data = data.join(vitaly)
    # ido = delete_stop_words_create_stems(data['words'])  # TODO: use this

    data = data.drop(columns=['words', 'tweet'])
    for i in range(10):
        current = data[data['user'] == i]
        zv.save_heatmap(current, i)
    # zv.pair_plot(data, data.columns)
    print()
