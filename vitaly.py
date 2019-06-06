import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import os
from os.path import isfile, join


#  break a tweet to its base features as selected beforehand
def process_sentence(tweets):
    result = pd.DataFrame()
    str_tweets = tweets.apply(lambda tweet: str(tweet))
    filtered = str_tweets.apply(lambda tweet: tweet[2:-1])
    filtered = filtered.apply(lambda tweet: re.sub(r"^\[[\"|\']", r'', tweet))
    filtered = filtered.apply(lambda tweet: re.sub(r"[\"|\']\]$", r'', tweet))
    result['has_link'] = filtered.apply(lambda tweet: 1 if re.search(r"http[s]?://[A-Za-z0-9\.\/]*\s*", tweet) is not None else 0)
    filtered = filtered.apply(lambda tweet: re.sub(r"http[s]?://[A-Za-z0-9\.\/]*\s*", r"", tweet))
    result['is_RT'] = filtered.apply(lambda tweet: 1 if re.search(r"^RT *@[\w]*:", tweet) is not None else 0)
    filtered = filtered.apply(lambda tweet: re.sub(r"^RT *@[\w]*:", r'', tweet))
    result['has_dots'] = filtered.apply(lambda tweet: 1 if 0 < tweet.count('.') else 0)
    result['has_tags'] = filtered.apply(lambda tweet: 1 if 0 < tweet.count('@') else 0)
    result['has_hash'] = filtered.apply(lambda tweet: 1 if 0 < tweet.count('#') else 0)
    result['has_exmark'] = filtered.apply(lambda tweet: 1 if 0 < tweet.count('!') else 0)
    result['has_qmark'] = filtered.apply(lambda tweet: 1 if 0 < tweet.count('?') else 0)
    filtered = filtered.apply(lambda tweet: re.sub(r"\s+", r' ', tweet))
    filtered = filtered.apply(lambda tweet: re.sub(r"[\.,]", r'', tweet))
    filtered = filtered.apply(lambda tweet: tweet.encode('ascii', 'ignore').decode('ascii'))
    result['words'] = filtered.apply(lambda tweet: tweet.split())
    return result
    # st = re.sub(r"^\[[\"|\']", r'', '')
    # st = re.sub(r"[\"|\']\]$", r'', st)
    # has_link = 1 if re.search(r"http[s]?://[A-Za-z0-9\.\/]*\s*", st) is not None else 0
    # st = re.sub(r"http[s]?://[A-Za-z0-9\.\/]*\s*", r"", st)
    # is_rt = 1 if re.search(r"^RT *@[\w]*:", st) is not None else 0
    # st = re.sub(r"^RT *@[\w]*:", r'', st)
    # has_dots = 1 if 0 < st.count('.') else 0
    # has_tags = 1 if 0 < st.count('@') else 0
    # has_hashs = 1 if 0 < st.count('#') else 0
    # has_exc = 1 if 0 < st.count('!') else 0
    # has_qmark = 1 if 0 < st.count('?') else 0
    # st = re.sub(r"\s+", r' ', st)
    # st = re.sub(r"[\.,]", r'', st)
    # st = st.encode('ascii', 'ignore').decode('ascii')
    # print(st.split())
    # return [is_rt, has_link, has_dots, has_exc, has_tags, has_hashs, has_qmark, st.split()]


if __name__ == '__main__':
    data = pd.DataFrame()
    for filename in os.listdir('tweets_data'):
        if filename.endswith('.csv'):
            data = data.append(pd.read_csv('tweets_data/' + filename))
    data['tweet'] = data['tweet'].apply(
        lambda w: w.encode(encoding='utf-8', errors='ignore'))

    # minimal preprocessing on the strings
    data['tweet'] = data['tweet'].apply(
        lambda w: w[2:len(w) - 2])  # remove [' and ']

    v = process_sentence(data['tweet'])

    vitaly = data['tweet'].apply(lambda x: np.array(
        process_sentence(x)))
    feature_names = ['is_rt', 'has_link', 'has_dots', 'has_exc', 'has_tags', 'has_hashs', \
    'has_qmark']
    for i in range(len(vitaly[0]) - 1):
        data[feature_names[i]] = vitaly[i]
    from four_c import delete_stop_words_create_stems
    mor = vitaly.apply(lambda x: x[-1])
    ido = delete_stop_words_create_stems(mor)
    print()

    #
    # features = ['tagged',
    #             'average number of words',
    #             'number of dots',
    #             'number of ,',
    #             'hashtags used',
    #             'retweet or not',
    #             'number of capitalized words',
    #             'number of !',
    #             'has link',
    #             'contains some keyword (Iraq, Afghanistan...)',
    #             'contains flags',
    #             'contains ...']
    # listed = [list(), list()]
    # path = 'tweets_data/'
    # for file in [f for f in listdir(path) if isfile(join(path, f))]:
    #     if file[-11:] == '_tweets.csv':
    #         user_tweets = pd.read_csv(join(path, file), sep=',')
    #         listed[0] += list(user_tweets['user'].values)
    #         listed[1] += list(user_tweets['tweet'].values)
    # total_data = pd.DataFrame(list(zip(listed[0], listed[1])), columns=['user', 'tweet'])
    # # total_data['tweet'] = total_data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))
    #
    # for i in range(0, 30000, 3000):
    #     words = process_sentence(total_data['tweet'][i])[-1]
