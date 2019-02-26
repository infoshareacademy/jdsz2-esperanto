import pandas as pd
import numpy as np

dataset = pd.read_csv('nyc-rolling-sales.csv')
print(dataset.head(0))

columns_to_model = ['BOROUGH', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE', 'RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'GROSS SQUARE FEET', 'YEAR BUILT' ]
columns_to_add_later = ['NEIGHBORHOOD', 'ZIP CODE']
X = dataset[columns_to_model]
y = dataset['SALE PRICE']

# korelacje do sprawdzenia:
# building class category i building class at present
#naighbours i zip code
#
X['YEAR BUILT'].replace({0: np.nan}, inplace=True)
#print(dataset['YEAR BUILT'].isna().any(), dataset['YEAR BUILT'].shape,dataset.shape)
#dataset.dropna(inplace=True)
#print(dataset['YEAR BUILT'].isna().any(), dataset['YEAR BUILT'].shape, dataset.shape)
X['GROSS SQUARE FEET'].replace({'0': np.nan, ' -  ': np.nan}, inplace=True)
X.dropna(inplace=True)
print(X)
#dataset['GROSS SQUARE FEET'].dropna(inplace=True)


for f_name in columns_to_model:
     print(f_name, ' ', dataset[f_name].unique())
#print(len(dataset['GROSS SQUARE FEET'].unique()))
# # TODO: zero year, gross square feet
#
# a = dataset['GROSS SQUARE FEET'].unique()
# print(a)
# print(dataset)
#
#
#
# # sprawdzamy wystepowanie NAN
#
# # print(X.head(0))
# # print(y)
