import pandas as pd


class ZivsProcessor:
    fill = {"median": lambda data: data.median(),
            "mode": lambda data: data.mode()[0]}

    correcter = {'delete': lambda data, feature, indices: data.drop(index=indices, inplace=True)}

    @staticmethod
    def categorize_features_by_values_amount(data, threshold=2):
        '''
        Can be used to convert a/b values to 0/1 values using threshold=2
        :param data: the data
        :param threshold: the MAX amount of different values we would like to split
        :return: the categorized data
        '''
        categorized_features = [col for col in data.columns if
                                len(data[col].value_counts()) <= threshold]
        return pd.get_dummies(data, columns=categorized_features, drop_first=True,
                              prefix_sep='$')

    @staticmethod
    def complete_missing_data(data, feature, filling="median"):
        '''
        Complete the missing values in the data by a certain method (median, mode, etc.), or a
        specific value
        :param data: data
        :param feature: feature to be completed
        :param filling: way of completing -- possible options are listed in class fill dictionary,
        can be used as a value
        :return: None, data is changed globally
        '''
        if filling in ZivsProcessor.fill.keys():
            data[feature].fillna(ZivsProcessor.fill[filling](data[feature]), inplace=True)
        else:
            data[feature].fillna(int(filling), inplace=True)

    @staticmethod
    def convert_rare_values(data, feature, threshold=10, new_value='Misc',
                            print_result=False):
        '''
        Convert the rare values of the feature to a new_value, by a given threshold
        :param data: the data
        :param feature: feature to convert from
        :param threshold: values will less appearances than this will become new_value
        :param new_value: the value to put in rare values
        :param print_result: default is false
        :return: The changed data
        '''
        title_names = (data[feature].value_counts() < threshold)
        data[feature] = data[feature].apply(
            lambda x: new_value if title_names.loc[x] == True else x)
        if print_result:
            print(data[feature].value_counts())
            print("-" * 10)
        return data

    @staticmethod
    def convert_features_with_label_encoder(data, features):
        '''
        Adds the converted feature as a new col with 'feature_Code'
        :param data: the data
        :param features: features to be converted
        :return: changed data
        '''
        from sklearn.preprocessing import LabelEncoder
        label = LabelEncoder()
        for feature in features:
            data[feature + '_Code'] = label.fit_transform(data[feature].fillna('0').astype(str))
            data = data.drop(columns=[feature])
        return data

    @staticmethod
    def correct_extreme_values(data, feature, min_value, max_value, correcting_method="delete"):
        indices = data[feature][(data[feature] > max_value) | (data[feature] < min_value)].index
        ZivsProcessor.correcter[correcting_method](data, feature, indices)
        return data
