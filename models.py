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
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# from pandas.tools.plotting import scatter_matrix


def random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 random_state=0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(np.mean(predicted == y_test))


def svm_f(X_train, X_test, y_train, y_test):
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
    # pd.read_csv()


def adaboost(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    # AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
    #                    learning_rate=1.0, n_estimators=100, random_state=0)
    predicted = clf.predict(X_test)
    print(np.mean(predicted == y_test))



def run_another_model(X_train, X_test, y_train, y_test):
    XGBClassifier()


def combine_models(X_train, X_test, y_train, y_test):
    # Machine Learning Algorithm (MLA) Selection and Initialization
    MLA = [
        # # Ensemble Methods
        # ensemble.AdaBoostClassifier(),
        # ensemble.BaggingClassifier(),
        # ensemble.ExtraTreesClassifier(),
        # ensemble.GradientBoostingClassifier(),
        # ensemble.RandomForestClassifier(),
        #
        # # Gaussian Processes
        # gaussian_process.GaussianProcessClassifier(),
        #
        # # GLM
        # linear_model.LogisticRegressionCV(),
        # linear_model.PassiveAggressiveClassifier(),
        # linear_model.RidgeClassifierCV(),
        # linear_model.SGDClassifier(),
        # linear_model.Perceptron(),

        # Navies Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        # Nearest Neighbor
        neighbors.KNeighborsClassifier(),

        # SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        # Trees
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        # Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),

        # xgboost: http://xgboost.readthedocs.io/en/latest/model.html
        XGBClassifier()
    ]
    # split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
    # note: this is an alternative to train_test_split
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3,
                                            train_size=.6,
                                            random_state=0)  # run model 10x with 60/30 split intentionally leaving out 10%

    # create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean',
                   'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD',
                   'MLA Time']
    MLA_compare = pd.DataFrame(columns=MLA_columns)

    # create table to compare MLA predictions
    MLA_predict = y_train

    X_train = X_train[:18167]
    y_train = y_train[:18167]

    # index through MLA and save performance to table
    row_index = 0
    for alg in MLA:
        # set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        # score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, X_train,
                                                    y_train, cv=cv_split)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results[
            'train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results[
            'test_score'].mean()
        # if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results[
                                                                    'test_score'].std() * 3  # let's know the worst that can happen!

        # save MLA predictions - see section 6 for usage
        alg.fit(X_train, y_train)
        MLA_predict[MLA_name] = alg.predict(X_train)

        row_index += 1

    # print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False,
                            inplace=True)
    MLA_compare
    # MLA_predict


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
    del X['Unnamed: 0']
    X_train, X_vt, y_train, y_vt = \
              train_test_split(X, y, test_size=0.4, random_state=42)

    # X_valid, X_test, y_valid, y_test = \
    #     train_test_split(X_vt, y_vt, test_size=0.5, random_state=42)
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
    # knn(X_train, X_test, y_train, y_test, X_vt, y_vt)
    combine_models(X_train[:18166], X_vt, y_train[:18166], y_vt)



