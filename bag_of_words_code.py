import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':
    data = pd.DataFrame()
    for filename in os.listdir('tweets_data'):
        if filename.endswith('.csv'):
            data = data.append(pd.read_csv('tweets_data/' + filename))
    data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))

    # minimal preprocessing on the strings
    data['tweet'] = data['tweet'].apply(lambda w: w[2:len(w) - 2])  # remove [' and ']

    # create BoW
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline

    labels = pd.Series(data['user'])
    data = data.drop(columns=['user'])

    # split to test
    X_train, X_test, y_train, y_test = \
        train_test_split(data, labels, test_size=0.33, random_state=42)

    X_train = X_train.fillna('0').astype(bytes)
    y_train = y_train.fillna(0).astype(float)

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf.fit(X_train['tweet'], y_train)
    predicted = text_clf.predict(X_test['tweet'])
    print(np.mean(predicted == y_test))