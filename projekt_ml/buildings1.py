import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics

dataset = pd.read_csv('nyc-rolling-sales.csv')

#przygotowanie datasetu do dalszych analiz:

columns_to_delete = ['Unnamed: 0', 'NEIGHBORHOOD', 'TAX CLASS AT PRESENT', 'BLOCK', 'LOT',
                     'SALE DATE', 'EASE-MENT', 'BUILDING CLASS AT PRESENT', 'ADDRESS',
                     'TOTAL UNITS', 'APARTMENT NUMBER', 'ZIP CODE', 'LAND SQUARE FEET']
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

print(X)
print(y)
