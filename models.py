import pandas as pd
import numpy as np
import regex as re
import os
from os.path import isfile, join
import vitaly
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer


def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=300, max_depth=14,
                                 random_state=0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(np.mean(predicted == y_test))


def svm(X_train, X_test, y_train, y_test):
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    predicted = clf.predict(X_test)
    print(np.mean(predicted == y_test))


def knn(X_train, X_test, y_train, y_test, X_valid, y_valid):
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(X_train, y_train)
    predicted = neigh.predict(X_test)
    print("test err:" +str(np.mean(predicted == y_test)))
    predicted_v = neigh.predict(X_valid)
    print("valid err:" + str(np.mean(predicted_v == y_valid)))


def adaboost(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    # AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
    #                    learning_rate=1.0, n_estimators=100, random_state=0)
    predicted = clf.predict(X_test)
    print(np.mean(predicted == y_test))


if __name__ == '__main__':
    data = pd.DataFrame()
    # for filename in os.listdir('tweets_data'):
    #     if filename.endswith('.csv'):
    #         data = data.append(pd.read_csv('tweets_data/' + filename))
    data = pd.read_csv('sets/data_with_user.csv')
    data['tweet'] = data['tweet'].apply(
        lambda w: w.encode(encoding='utf-8', errors='ignore'))
    del data['words']
    del data['tweet']
    X = data
    y = data['user']
    del X['user']
    X_train, X_vt, y_train, y_vt = \
              train_test_split(X, y, test_size=0.4, random_state=42)

    X_valid, X_test, y_valid, y_test = \
        train_test_split(X_vt, y_vt, test_size=0.5, random_state=42)
    # data_features = vitaly.process_sentence(data['tweet'])
    # vectorizer = CountVectorizer()
    # combined_tweets = ''
    # for tweet in data_features['words']:
    #     for word in tweet:
    #         combined_tweets = combined_tweets + ' ' + word
    # words_features = vectorizer.fit_transform(combined_tweets)
    # words_features = vectorizer.fit_transform(data['tweet'])

    # del data_features['words']
    # X_train, X_test, y_train, y_test = \
    #     train_test_split(data_features, data['user'], test_size=0.33,
    #                      random_state=42)
    knn(X_train, X_test, y_train, y_test, X_valid, y_valid)



