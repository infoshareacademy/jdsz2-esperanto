import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn import svm


def main():
    pd.set_option('display.max_columns',10)
    df = pd.read_csv('Concrete_Data.csv',quotechar='"', decimal=',', skipinitialspace=True)
    df.to_csv('Concrete_Corr', sep=',', encoding='utf-8', quotechar='"', decimal='.')

    columns_to_model = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
                        'Coarse Aggregate', 'Fine Aggregate', 'Age']

    for row in df.itertuples():
        c = math.floor(row._9 / 5)
        df.at[row.Index, 'Concrete compressive strength'] = c * 5
        print(row)

    X = df[columns_to_model]
    y = df['Concrete compressive strength']

    print(X, y, df['Age'])


if __name__ == '__main__':
    main()