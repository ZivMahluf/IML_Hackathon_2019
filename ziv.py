import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import zero_one_loss

TYPE_OF_DATA = 'train'

if __name__ == '__main__':
    data = pd.DataFrame()
    if TYPE_OF_DATA == 'train':
        for filename in os.listdir('tweets_data'):
            if filename.endswith('.csv'):
                file_data = pd.read_csv('tweets_data/' + filename)
                file_data['user'].fillna(file_data['user'].mode(dropna=True))  # use only for traindata
                file_data['tweet'] = file_data['tweet'].dropna()
                data = data.append(file_data)
    else:

    data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))

    # minimal preprocessing on the strings
    data['tweet'] = data['tweet'].apply(lambda w: w[2:len(w) - 2])  # remove [' and ']

    # ---------- COMPLETING STAGE -----------
    #  completing - in train - correct label by given file. in train - delete if tweet is nan/null,
    #  in test - guess.

    # code for train data


    # code for test data