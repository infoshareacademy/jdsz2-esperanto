import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt

dataset = pd.read_csv('nyc-rolling-sales.csv')

#przygotowanie datasetu do dalszych analiz:

columns_to_delete = ['Unnamed: 0', 'NEIGHBORHOOD', 'TAX CLASS AT PRESENT', 'BLOCK', 'LOT',
                     'SALE DATE', 'EASE-MENT', 'BUILDING CLASS AT PRESENT', 'ADDRESS',
                     'TOTAL UNITS', 'APARTMENT NUMBER', 'ZIP CODE', 'LAND SQUARE FEET']
columns_to_regression = ['BOROUGH', 'TAX CLASS AT TIME OF SALE', 'RESIDENTIAL UNITS',
                         'COMMERCIAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT']
columns_to_model = ['BOROUGH', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT TIME OF SALE',
                    'BUILDING CLASS AT TIME OF SALE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS',
                    'GROSS SQUARE FEET', 'YEAR BUILT']
columns_to_add_later = ['NEIGHBORHOOD', 'ZIP CODE']

dataset.drop(columns=columns_to_delete, axis=1, inplace=True)
dataset['YEAR BUILT'].replace({0: None}, inplace=True)
dataset['GROSS SQUARE FEET'].replace({0: None, ' -  ': None}, inplace=True)
dataset['SALE PRICE'].replace({' -  ': None}, inplace=True)
dataset.dropna(inplace=True)
pd.set_option('display.max_columns',8)

# oczyszczone X i y do analiz:
X = dataset[columns_to_model]
y = dataset['SALE PRICE']
X_regr_lin = X.iloc[:,-2:]
#print(X_regr_lin)
#print(y)

# regresja liniowa:
X_train,X_test,y_train,y_test = train_test_split(X_regr_lin,y,test_size=0.25)
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

