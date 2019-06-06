import pandas as pd
import numpy as np
import regex as re
import os
from os.path import isfile, join


def process_sentence(tweets):
    open_pattern = re.compile(r"^\[[\"|\']")
    close_pattern = re.compile(r"[\"|\']\]$")
    link_pattern = re.compile(r"http[s]?://[A-Za-z0-9\.\/]*\s*")
    rt_pattern = re.compile(r"RT *@([\w]*):")
    spaces_pattern = re.compile(r"\s+")
    separators_pattern = re.compile(r"[\.,\"]")

    result = pd.DataFrame()
    result['num_of_non_ascii_characters'] = tweets.apply(lambda tweet: sum(127 < c for c in tweet))
    str_tweets = tweets.apply(lambda tweet: str(''.join([chr(c) for c in tweet if c <= 127])))
    filtered = str_tweets.apply(lambda tweet: open_pattern.sub(r'', tweet))
    filtered = filtered.apply(lambda tweet: close_pattern.sub(r'', tweet))
    result['has_link'] = filtered.apply(lambda tweet: 1 if link_pattern.search(tweet) is not None else 0)
    filtered = filtered.apply(lambda tweet: link_pattern.sub(r'', tweet))
    result['is_RT'] = filtered.apply(lambda tweet: 1 if rt_pattern.search(tweet) is not None else 0)
    filtered = filtered.apply(lambda tweet: rt_pattern.sub(r'', tweet))
    result['num_dots'] = filtered.apply(lambda tweet: tweet.count('.'))
    result['num_tags'] = filtered.apply(lambda tweet: tweet.count('@'))
    result['num_hashes'] = filtered.apply(lambda tweet: tweet.count('#'))
    result['num_exmarks'] = filtered.apply(lambda tweet: tweet.count('!'))
    result['num_qmarks'] = filtered.apply(lambda tweet: tweet.count('?'))
    filtered = filtered.apply(lambda tweet: spaces_pattern.sub(r' ', tweet))
    filtered = filtered.apply(lambda tweet: separators_pattern.sub(r'', tweet))
    filtered = filtered.apply(lambda tweet: tweet.encode('ascii', 'ignore').decode('ascii'))
    result['words'] = filtered.apply(lambda tweet: tweet.strip('\s').split())
    result['uppercase_words_ratio'] = result['words'].apply(lambda word_lst: sum(list(map(str.isupper, word_lst))))
    result['lowercase_words_ratio'] = result['words'].apply(lambda word_lst: sum(list(map(str.islower, word_lst))))
    result['words_per_tweet'] = result['words'].apply(lambda word_lst: len(word_lst))

    return result


def f1():
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
    feature_names = ['is_rt', 'has_link', 'has_dots', 'has_exc', 'has_tags', 'has_hashs', 'has_qmark']
    for i in range(len(vitaly[0]) - 1):
        data[feature_names[i]] = vitaly[i]
    from four_c import delete_stop_words_create_stems
    mor = vitaly.apply(lambda x: x[-1])
    ido = delete_stop_words_create_stems(mor)
    print()


def f2():
    features = ['tagged',
                'average number of words',
                'number of dots',
                'number of ,',
                'hashtags used',
                'retweet or not',
                'number of capitalized words',
                'number of !',
                'has link',
                'contains some keyword (Iraq, Afghanistan...)',
                'contains flags',
                'contains ...']
    listed = [list(), list()]
    path = 'tweets_data/'
    for file in [f for f in os.listdir(path) if isfile(join(path, f))]:
        if file[-11:] == '_tweets.csv':
            user_tweets = pd.read_csv(join(path, file), sep=',')
            listed[0] += list(user_tweets['user'].values)
            listed[1] += list(user_tweets['tweet'].values)
    data = pd.DataFrame(list(zip(listed[0], listed[1])), columns=['user', 'tweet'])
    data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))
    all_data = data.join(process_sentence(data['tweet']))
    all_data = pd.get_dummies(all_data, columns=['user'])
    print()


def f3():
    pass


if __name__ == '__main__':
    # f1()
    f2()
    # f3()
