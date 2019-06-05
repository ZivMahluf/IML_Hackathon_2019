import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ZivsVisualizer:

    @staticmethod
    def save_heatmap(data):
        '''
        Save the heatmap. data is assumed to have labels as a column in it.
        :param data:
        :return:
        '''
        _, ax = plt.subplots(figsize=(14, 12))
        colormap = sns.diverging_palette(220, 10, as_cmap=True)

        _ = sns.heatmap(
            data.corr(),
            cmap=colormap,
            square=True,
            cbar_kws={'shrink': .9},
            ax=ax,
            annot=True,
            linewidths=0.1, vmax=1.0, linecolor='white',
            annot_kws={'fontsize': 12}
        )

        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        plt.savefig("images/heatmap.png", type='png')

    @staticmethod
    def pair_plot(data, features):
        '''
        Plot a figure with a pair of each feature with all the other features.
        :param data: the data
        :param features: the features
        :return: None
        '''
        sns.set()
        sns.pairplot(data[features], size=2.5)
        plt.savefig('images/pair_plot.png', format='png')

    @staticmethod
    def print_and_return_cols_with_null(data, amount_printed=20):
        '''
        Prints the PERCENTAGE of missing values for the amount_printed columns with the highest
        missing ratio
        :param data: the data
        :param amount_printed: (default=20) amount of columns to print.
        :return: list of column names
        '''
        all_data_na = (data.isnull().sum() / len(data)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(
            ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        print(missing_data.head(amount_printed))

        null_data = data.isnull().sum()
        return null_data[null_data != 0].index.tolist()

    @staticmethod
    def save_classification_graphs(data, labels):
        '''
        Save scatter graphs for all features, where each point in the graph is marked by the label
        of the feature.
        :return: None
        '''
        fig, ax = plt.subplots(1, 1)
        pos = np.where(labels == 1)
        neg = np.where(labels == 0)
        cdict = {0: 'red', 1: 'green'}
        for feature in data.columns:
            ax.scatter(data[feature].iloc[neg], neg, c=cdict[0], s=20, label='dead')
            ax.scatter(data[feature].iloc[pos], pos, c=cdict[1], s=20, label='alive')
            ax.set_ylabel('labels')
            ax.set_xlabel(feature)
            ax.set_title(feature + ' by label')
            ax.legend()
            plt.savefig('images/' + feature + '.png', format='png')
            plt.cla()

    @staticmethod
    def save_regression_graphs(data, labels, y_axis_title="Labels"):
        '''
        Save the images of all the features by labels graphs.
        :param data: the data, with or without the labels
        :param labels: the labels
        :param y_axis_title: The name to be displayed in the graph's y axis
        :return: None
        '''
        fig, ax = plt.subplots(1, 1)
        for feature in data.columns:
            ax.scatter(data[feature].fillna('0').values, labels, label=feature)
            ax.set_ylabel(y_axis_title)
            ax.set_xlabel(feature)
            ax.set_title(feature + ' by ' + y_axis_title)
            ax.legend()
            plt.savefig('corr_feature_images/' + feature + '.png', format='png')
            plt.cla()

    @staticmethod
    def percentage_of_ones_for_features(data, labels, threshold=2):
        '''
        Save a bar graph, where the x_axis is the feature name, and the y graph is the percentage
        of y=1 for the value of the feature
        For each binary feature, we will have 2 bars, one for the feature=1 and one for the
        feature=0
        :param threshold: The MAX amount of different values we allow the feature to have
        :return: None
        '''

        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """

            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0, 'right': 1, 'left': -1}

            for rect in rects:
                height = rect.get_height()
                plt.annotate('{}'.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                             textcoords="offset points",  # in both directions
                             ha=ha[xpos], va='bottom')

        plt.rcParams.update({'font.size': 8})
        fig, ax = plt.subplots(figsize=(20, 6))
        for feature in data.columns:
            diff_values = data[feature].value_counts()
            if len(diff_values) <= threshold:
                for value in diff_values.index:
                    # will contain 1 when data is 'value' AND label is 1
                    condition_met = np.where(data[feature] == value, labels, 0)
                    # plot the percentage of feature=value and label=1
                    rects = ax.bar(feature + '=' + str(value),
                                   round(condition_met.sum() / len(condition_met), 2),
                                   label=feature + '=' + str(value))
                    autolabel(rects)

        # rotate x axis test so it'll be seen
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        plt.ylabel('percentage of y=1 for ')
        plt.legend(loc='best', frameon=False)
        plt.savefig('images/percentage_of_features.png', format='png')
        plt.cla()

    @staticmethod
    def print_label_percentage(data, label_feature_name, labels, percentage=True):
        '''
        One of the most important functions in here.
        For every feature in the data, it prints the percentage of each value for both
        Target/Label=1 and 0. Will show you clearly which feature VALUES are related to the target.

        Note: makes local copy of data. if data is too big you can change this function to use it
        w/o a copy by sending the data with target features as part of it.
        :param data: the data
        :param label_feature_name: the name of the target feature/column
        :param labels: the labels
        :param percentage: (default:True) print percentage, if you want AMOUNT, use false.
        :return: None
        '''
        loc_data = data.copy(deep=True)
        loc_data[label_feature_name] = labels
        for col in loc_data.columns:
            if loc_data[col].dtype != 'float64' and col != label_feature_name:
                print('Correlation between ' + label_feature_name + ' and ' + col)
                if percentage:
                    print(loc_data[[col, label_feature_name]].groupby(col, as_index=False).mean())
                else:
                    print(pd.crosstab(loc_data[col], loc_data[label_feature_name]))
                print('-' * 10, '\n')
