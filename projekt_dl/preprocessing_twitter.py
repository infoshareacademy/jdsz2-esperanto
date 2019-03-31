import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


## --------------------- Import dataset ---------------------
#http://help.sentiment140.com/for-students/

pd.set_option('display.max_columns',10)
train_dataset = pd.read_csv('trainingandtestdata/train.csv',quotechar='"',sep=',',encoding='ISO-8859-1')
test_dataset = pd.read_csv('trainingandtestdata/test.csv',quotechar='"',sep=',',encoding='ISO-8859-1')

## --------------------- Define target and text ---------------------
#print(train_dataset.head())
X_train = train_dataset.iloc[:,5] # CORPUS
y_train = train_dataset.iloc[:,0] # MANY TO ONE

X_test = test_dataset.iloc[:,5] # CORPUS
y_test = test_dataset.iloc[:,0] # MANY TO ONE

## --------------------- Create words dictionary ---------------------
corpus = X_train
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
#print(X.toarray())


