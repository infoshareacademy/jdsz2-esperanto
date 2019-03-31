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
from xgboost import XGBRegressor
from sklearn import svm
import warnings

warnings.filterwarnings("ignore")

## --------------------- import dataset ---------------------
## https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
pd.set_option('display.max_columns',10)
dataset = pd.read_csv('Concrete_Data.csv',quotechar='"', decimal=',', skipinitialspace=True)
dataset.to_csv('Concrete_Corr.csv', sep=',', encoding='utf-8', quotechar='"', decimal='.')

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
#print(max(y),min(y)) =   82.6 <-> 2.33


## ------------- feature dependencies on target -------------
for i in columns_to_model:
    plt.scatter(X[i], y)
    plt.title(i)
#    plt.show()


## -------- dataset training and linear regression ----------
# podzial datasetu na dane treningowe i testowe
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=101)


## ------------- cross validations MAE / MSE --------------
my_imputer = Imputer()
train_X = my_imputer.fit_transform(X_train)
test_X = my_imputer.transform(X_test)

kfold = KFold(n_splits=20, random_state=11)

scorer_MAE = make_scorer(mean_absolute_error)
scorer_MSE = make_scorer(mean_squared_error)


print('________________________________________________________')
print('                    LinearRegression                    ')
print('________________________________________________________')

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

print('                +++ Cross Validation  +++               ')
results_MAE = cross_val_score(regr, X_train, y_train, cv=kfold, scoring=scorer_MAE)
results_MSE = cross_val_score(regr, X_train, y_train, cv=kfold, scoring=scorer_MSE)

print('Mean_MAE:  {}'.format((results_MAE.mean())))
print('Mean_MSE:  {}'.format((results_MSE.mean())))

print('________________________________________________________')
print('                  DecisionTreeRegressor                 ')
print('________________________________________________________')
for max_features in [1,2,3,4,5,6,7,8]:
    dtree = DecisionTreeRegressor(max_features=max_features, random_state=None, max_depth=None,
                                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0)
    dtree.fit(X_train, y_train)
    y_pred_tree = dtree.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_tree)
    r2 = metrics.r2_score(y_test, y_pred_tree)
    print('For max features: {}    Mean Absolute Error:  {}    R^2:  {}'.format(max_features, mae, r2))

print('                +++ Cross Validation  +++               ')
results_MAE = cross_val_score(dtree, X_train, y_train, cv=kfold, scoring=scorer_MAE)
results_MSE = cross_val_score(dtree, X_train, y_train, cv=kfold, scoring=scorer_MSE)

print('Mean_MAE:  {}'.format((results_MAE.mean())))
print('Mean_MSE:  {}'.format((results_MSE.mean())))


print('________________________________________________________')
print('                         XGBoost                        ')
print('________________________________________________________')


clf_xgbr = XGBRegressor()
clf_xgbr.fit(X_train, y_train, verbose=False)

# Dokonujemy predykcji dla danych testowych przy użyciu XGBClassifier.
y_pred_xgb = clf_xgbr.predict(X_test)

print("Mean Absolute Error dla danych testowych : " + str(mean_absolute_error(y_pred_xgb, y_test)))

results_MAE = cross_val_score(clf_xgbr, X_train, y_train, cv=kfold, scoring=scorer_MAE)
results_MSE = cross_val_score(clf_xgbr, X_train, y_train, cv=kfold, scoring=scorer_MSE)

# Wypisujemy wynik cross-validacji.
print('Mean_MAE:  {}'.format((results_MAE.mean())))
print('Mean_MSE:  {}'.format((results_MSE.mean())))


print('________________________________________________________')
print('                           SVR                          ')
print('________________________________________________________')

svr_lin = svm.SVR(kernel='linear')
svr_lin.fit(X_train, y_train)
y_lin = svr_lin.predict(X_test)

print('R^2:', metrics.r2_score(y_test, y_lin))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_lin))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_lin))
print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_lin))

svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X_train, y_train)
y_rbf = svr_rbf.predict(X_test)

print('R^2:', metrics.r2_score(y_test, y_rbf))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_rbf))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_rbf))
print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_rbf))

