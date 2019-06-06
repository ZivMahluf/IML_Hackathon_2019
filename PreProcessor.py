import os
import nltk
import regex as re
import pandas as pd
from four_c import delete_stop_words_create_stems
from visualize_data import ZivsVisualizer
from collections import defaultdict


class PreProcessor:
    Trump = {'great', 'Border', 'President', 'people', 'Democrats', 'Trump',
             'Country', 'US', 'Wall', 'Fake', 'Great', 'Mueller', 'China',
             'States', 'United'}

    Joe_Biden = {'Biden', 'VP', 'President', 'Joe', 'country', 'Obama',
                 'America', 'middle', 'Vice'}

    Conan_Obrian = {'show', 'know', 'Trump', 'say', 'dont'}

    Ellen_Degeneres = {'Happy', 'birthday', 'love', 'show', '#GameofGames',
                       '#ThanksSponsor', 'gonna', 'full', 'hope'}

    Kim_Kardeshian = {'PST', '12PM', '#KKWBEAUTY', 'Shop', '@kkwbeauty',
                      '@KimKardashian', 'available', 'Crme', 'Classic', 'Lip',
                      'Happy', 'Pop-Up', '#KUWTK', 'love', 'Lipstick', 'KKW',
                      'Collection'}

    Lebrom_James = {'@KingJames', 'u', '#StriveForGreatness', 'Conqrats',
                    'game', 'S/O', '#Striveforgreatne', 'Man', 'homie', 'Love',
                    'guys'}

    Lady_GaGa = {'Gaga', 'Lady', 'love', 'u', '@ladygaga', 'Thank', 'Tony',
                 'world', '#JOANNE', 'beautiful', 'music'}

    Cristiano_Ronaldo = {'@Cristiano', 'game', 'win', 'Thank', 'eat', 'de',
                         'team', 'asked:', 'Check', 'support', 'fans', 'CR7'}

    Jimmy_Kimmel = {'@RealDonaldTrump', '@IamGuillermo', 'great', '@jimmykimmel',
                    'show', '@TheCousinSal', 'ever'}

    Schwarzeneger = {'@Schwarzenegger', 'Arnold', 'great', 'fantastic', 'love',
                     'proud', '@ArnoldSports', 'gerrymandering'}

    Sets_mapping = [Trump, Joe_Biden, Conan_Obrian, Ellen_Degeneres, Kim_Kardeshian, Lebrom_James,
                    Lady_GaGa, Cristiano_Ronaldo, Jimmy_Kimmel, Schwarzeneger]

    @staticmethod
    def __save_frequent_words_per_user(data):
        nltk.download('stopwords', quiet=True)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        stop_words = stop_words.union({'&amp;', 'new', 'one', 'time'})
        user_common_word_set = dict()
        for i in range(10):
            count_vect = defaultdict(int)
            user_indices = data[data['user_' + str(i)] == 1].index
            words = data.iloc[user_indices]['words']
            for word_list in words:
                for x in word_list:
                    if x.lower() not in stop_words:  # remove stopwords from 'words' column
                        count_vect[x] += 1
            v = sorted(count_vect.items(), key=lambda kv: kv[1], reverse=True)  # sort ascending
            user_common_word_set[i] = v[0:100]

        # saves common word set to an external file, to use in manual feature selection
        for j in range(10):
            f = open('sets/set_user_' + str(j), 'w')
            f.write(str(user_common_word_set[j]))
            f.close()

    @staticmethod
    def __add_word_columns(data):
        set_features = pd.DataFrame(columns=['set_' + str(k) + '_score' for k in range(10)])
        for word_list_index, word_list in data['words'].iteritems():
            score = defaultdict(float)
            for word in word_list:
                for g in range(10):
                    if word in PreProcessor.Sets_mapping[g]:
                        score[g] += 1
            set_features = set_features.append(score, ignore_index=True)
            print(word_list_index)

        data = pd.concat([data, set_features], axis=1)
        data = data.drop(columns=['set_' + str(k) + '_score' for k in range(10)])
        data = data.fillna(0)
        for w in data.columns:
            if str(w).find('set') != -1:
                data[w] = data[w].fillna(0).astype(int)
        return data

    @staticmethod
    def __read_training():
        data = pd.DataFrame()
        for filename in os.listdir('tweets_data'):
            if filename.endswith('_tweets.csv'):
                file_data = pd.read_csv('tweets_data/' + filename)
                file_data['user'].fillna(
                    file_data['user'].mode(dropna=True))  # use only for traindata
                file_data['tweet'].dropna(inplace=True)
                data = data.append(file_data, ignore_index=True)
        data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))
        return data

    @staticmethod
    def __read_test(data):
        new_data = pd.DataFrame()
        new_data['tweet'] = pd.Series(data)
        return new_data

    @staticmethod
    def __extract_tweets(tweets):
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

    @staticmethod
    def pre_process(data_to_read=None):
        if data_to_read is None:
            data = PreProcessor.__read_training()
        else:
            data = PreProcessor.__read_test(data_to_read)
        extracted_data = pd.concat([data, PreProcessor.__extract_tweets(data['tweet'])], axis=1)
        labels = pd.Series(data['user'])
        ZivsVisualizer.save_heatmap(pd.get_dummies(extracted_data, columns=['user']))
        return PreProcessor.__add_word_columns(extracted_data).drop(columns=['user']), labels
