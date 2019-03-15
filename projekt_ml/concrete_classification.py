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
from xgboost import XGBRegressor


def main():
    pd.set_option('display.max_columns', 10)
    df = pd.read_csv('Concrete_Corr', decimal='.', skipinitialspace=True)

    columns_to_model = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
                        'Coarse Aggregate', 'Fine Aggregate', 'Age']

    for row in df.itertuples():
        c = math.floor(row._10 / 5)
        df.at[row.Index, 'Concrete compressive strength'] = c * 5

    X = df[columns_to_model]
    y = df['Concrete compressive strength']

    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25, random_state=11)

    clf = XGBRegressor()
    clf.fit(X_train, y_train, verbose=False)
    predictions = clf.predict(X_test)
    print("Mean Absolute Error : " + str(mean_absolute_error(predictions, y_test)))
    scorer = make_scorer(mean_absolute_error)
    kfold = KFold(n_splits=5, random_state=11)
    results = cross_val_score(clf, X_train, y_train, cv=kfold, scoring=scorer)
    print(np.mean(results))


if __name__ == '__main__':
    main()
