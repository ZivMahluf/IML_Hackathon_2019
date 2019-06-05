import twitter as twitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
import seaborn as sns

from visualize_data import ZivsVisualizer as zv
from process_data import ZivsProcessor as zp
from itertools import product
from sklearn import tree
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

import os

if __name__ == '__main__':
    data = pd.DataFrame()
    for filename in os.listdir('tweets_data'):
        if filename.endswith('.csv'):
            data = data.append(pd.read_csv('tweets_data/' + filename))
    data['tweet'] = data['tweet'].apply(lambda w: w.encode(encoding='utf-8', errors='ignore'))

    print()
