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


def delete_stop_words_create_stems(data_tweet):
    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()
    new_data_tweet = []
    for tweet in data_tweet:
        new_tweet = []
        for word in tweet:
            if word not in stop_words:
                new_tweet.append(porter.stem(word))
        new_data_tweet.append(new_tweet)
    return new_data_tweet


