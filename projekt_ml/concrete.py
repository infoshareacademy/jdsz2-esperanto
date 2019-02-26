import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics


pd.set_option('display.max_columns',10)
dataset = pd.read_csv('Concrete_Data.csv')
#print(dataset)
#print('X\n',dataset.isnull().any())
columns_to_model = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
                    'Coarse Aggregate', 'Fine Aggregate', 'Age']
X = dataset[columns_to_model]
y = dataset['Concrete compressive strength']
#print(y.head())


# for i in columns_to_model:
#     plt.scatter(X[i], y)
#     plt.title(i)
#     plt.show()


dataset.astype(float)
print(dataset.dtypes)
# podzial datasetu na dane treningowe i testowe
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=101)
#
# # regresja liniowa:
# regr = linear_model.LinearRegression(normalize=True)
# regr.fit(X_train, y_train)
# y_pred = regr.predict(X_test)
#
# # The coefficient:
# print('Wyraz wolny:', regr.intercept_)
# print('Wspolczynniki:', regr.coef_)
# print('R^2:', metrics.r2_score(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred))
