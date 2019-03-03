import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error, make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

## --------------------- import dataset ---------------------
## https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
pd.set_option('display.max_columns',10)
dataset = pd.read_csv('Concrete_Data.csv',quotechar='"', decimal=',', skipinitialspace=True)
dataset.to_csv('Concrete_Corr', sep=',', encoding='utf-8', quotechar='"', decimal='.')


## Features -- quantitative -- kg in a m3 mixture -- Input Variables:
# 1-Cement, 2-Blast Furnace Slag, 3-Fly Ash, 4-Water, 5-Superplasticizer, 6-Coarse Aggregate, 7-Fine Aggregate
# 8-Age -- quantitative -- Day (1~365) -- Input Variable

## Concrete compressive strength -- quantitative -- MPa -- Output Variable


## --------------- define features and target ---------------
#print('X\n',dataset.isnull().any())
columns_to_model = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
                    'Coarse Aggregate', 'Fine Aggregate', 'Age']
X = dataset[columns_to_model]
y = dataset['Concrete compressive strength']
#print(y.head())


## ------------- feature dependencies on target -------------
for i in columns_to_model:
    plt.scatter(X[i], y)
    plt.title(i)
#    plt.show()


## ------------ dataset training and regression -------------
# podzial datasetu na dane treningowe i testowe
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=101)

# regresja liniowa:
regr = linear_model.LinearRegression(normalize=True)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# The coefficient:
print('Wyraz wolny:', regr.intercept_)
print('Wspolczynniki:', regr.coef_)
print('R^2:', metrics.r2_score(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred))


## ------------- cross validations MAE / MSE --------------
my_imputer = Imputer()
train_X = my_imputer.fit_transform(X_train)
test_X = my_imputer.transform(X_test)

kfold = KFold(n_splits=20, random_state=11)

scorer_MAE = make_scorer(mean_absolute_error)
results = cross_val_score(regr, X_train, y_train, cv=kfold, scoring=scorer_MAE)
print('==== Min_MAE:{}, Max_MAE:{} ===='.format(min(results), max(results)))

scorer_MSE = make_scorer(mean_squared_error)
results = cross_val_score(regr, X_train, y_train, cv=kfold, scoring=scorer_MSE)
print('==== Min_MSE:{}, Max_MSE:{} ===='.format(min(results), max(results)))