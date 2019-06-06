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


def knn(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(X_train, y_train)
    predicted = neigh.predict(X_test)
    print(np.mean(predicted == y_test))


def adaboost(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    # AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
    #                    learning_rate=1.0, n_estimators=100, random_state=0)
    predicted = clf.predict(X_test)
    print(np.mean(predicted == y_test))


if __name__ == '__main__':
    data = pd.DataFrame()
    for filename in os.listdir('tweets_data'):
        if filename.endswith('.csv'):
            data = data.append(pd.read_csv('tweets_data/' + filename))
    data['tweet'] = data['tweet'].apply(
        lambda w: w.encode(encoding='utf-8', errors='ignore'))
    data_features = vitaly.process_sentence(data['tweet'])
    del data_features['words']
    X_train, X_test, y_train, y_test = \
        train_test_split(data_features, data['user'], test_size=0.33,
                         random_state=42)
    adaboost(X_train, X_test, y_train, y_test)

