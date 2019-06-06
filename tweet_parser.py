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

import os

if __name__ == '__main__':
    data = pd.DataFrame()
    for filename in os.listdir('tweets_data'):
        if filename.endswith('.csv'):
            data = data.append(pd.read_csv('tweets_data/' + filename))
    data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))

    # minimal preprocessing on the strings
    data['tweet'] = data['tweet'].apply(lambda w: w[2:len(w)-2])  # remove [' and ']

    # create BoW
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline

    labels = pd.Series(data['user'])
    data = data.drop(columns=['user'])

    # split to test
    X_train, X_test, y_train, y_test = \
        train_test_split(data, labels, test_size = 0.33, random_state = 42)

    # text_clf = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultinomialNB()),
    # ])

    X_train = X_train.fillna('0').astype(bytes)
    y_train = y_train.fillna(0).astype(float)

    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(X_train['tweet'])
    #
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #
    # clf = MultinomialNB().fit(X_train_tfidf, y_train)
    # predicted = clf.predict(X_test)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(X_train['tweet'], y_train)
    predicted = text_clf.predict(X_test['tweet'])
    print(np.mean(predicted == y_test))

    import featuretools

    # check which features you get with feature tools

    # print()
