import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
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
    df['Concrete compressive strength'] = df['Concrete compressive strength'].astype(str)

    for row in df.itertuples():
        df.at[row.Index, 'Concrete compressive strength'] = 'M' + df.at[row.Index, 'Concrete compressive strength']

    X = df[columns_to_model]
    y = df['Concrete compressive strength']

    X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25, random_state=11)

    clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0, min_samples_leaf=2, criterion="gini")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)

    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print('Accuracy :', score)

if __name__ == '__main__':
    main()
