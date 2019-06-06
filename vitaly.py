import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from os import listdir
from os.path import isfile, join


#  break a tweet to its base features as selected beforehand
def process_sentence(tweet):
    st = re.sub(r"^\[[\"|\']", r'', tweet)
    st = re.sub(r"[\"|\']\]$", r'', st)
    has_link = 1 if re.search(r"http[s]?://[A-Za-z0-9\.\/]*\s*", st) is not None else 0
    st = re.sub(r"http[s]?://[A-Za-z0-9\.\/]*\s*", r"", st)
    is_rt = 1 if re.search(r"^RT *@[\w]*:", st) is not None else 0
    st = re.sub(r"^RT *@[\w]*:", r'', st)
    has_dots = 1 if 0 < st.count('.') else 0
    has_tags = 1 if 0 < st.count('@') else 0
    has_hashs = 1 if 0 < st.count('#') else 0
    has_exc = 1 if 0 < st.count('!') else 0
    has_qmark = 1 if 0 < st.count('?') else 0
    st = re.sub(r"\s+", r' ', st)
    st = re.sub(r"[\.,]", r'', st)
    st = st.encode('ascii', 'ignore').decode('ascii')
    print(st.split())
    return [is_rt, has_link, has_dots, has_exc, has_tags, has_hashs, has_qmark, st.split()]
    # print(tweet, end='\n\n')
    # tweet_structure = re.compile('\[[\'|\"](RT *@\w*:)? *([#@]?\w[\w\"\',]*)? *(http://[a-z\.A-Z0-9/]*)?.*[\'|\"]\]')
    # m = tweet_structure.match(tweet)
    # if m is not None:
    #     print(m.group(0))
    #     print(m.group(1))
    #     print(m.group(2))
    #     print(m.group(3))
    # else:
    #     print(tweet)
    # print()

    #  finding out if it's a retweet
    # rt = re.search('RT @(\w*):', tweet)
    # if rt is not None:
    #     #  if so - printing whose tweet is retweeted
    #     print(rt.group(1))
    # #  decomposing the tweet to its words
    # words_pattern = re.compile('(\w[\w\']*)')
    # words = words_pattern.findall(tweet)
    # print(words)
    # print()


if __name__ == '__main__':
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
    for file in [f for f in listdir(path) if isfile(join(path, f))]:
        if file[-11:] == '_tweets.csv':
            user_tweets = pd.read_csv(join(path, file), sep=',')
            listed[0] += list(user_tweets['user'].values)
            listed[1] += list(user_tweets['tweet'].values)
    total_data = pd.DataFrame(list(zip(listed[0], listed[1])), columns=['user', 'tweet'])
    # total_data['tweet'] = total_data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))
    for i in range(0, 30000, 3000):
        process_sentence(total_data['tweet'][i])
