import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv('forestfires.csv')
    X = df[['temp', 'wind', 'rain']]
    y = df['area']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    to_predict1 = np.array([18, 2.7, 0]).reshape(1, -1)   # 9.7 vs 0.36
    to_predict2 = np.array([20.3, 4.9, 0]).reshape(1, -1)   # 15.9 vs 4.53
    to_predict3 = np.array([24.1, 4.5, 0]).reshape(1, -1)   # 0 vs 20.48
    single_pred1 = clf.predict(to_predict1)
    single_pred2 = clf.predict(to_predict2)
    single_pred3 = clf.predict(to_predict3)
    print(single_pred1)
    print(single_pred2)
    print(single_pred3)
    # 9, 9, jul, tue, 85.8, 48.3, 313.4, 3.9, 18, 42, 2.7, 0, 0.36
    # 1, 4, aug, sat, 90.2, 96.9, 624.2, 8.9, 20.3, 39, 4.9, 0, 4.53
    # 2, 4, sep, sat, 92.5, 121.1, 674.4, 8.6, 24.1, 29, 4.5, 0, 0
    # X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area

    print('Wyraz wolny:', clf.intercept_)
    print('Wspolczynniki:', clf.coef_)
    print('R^2:', metrics.r2_score(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Median Absolute Error:', metrics.median_absolute_error(y_test, y_pred))


if __name__ == '__main__':
    main()
