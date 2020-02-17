'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

# !/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('Exercise 1.a')
    print('------------')
    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))


if __name__ == "__main__":

    # read data from file using pandas
    train_file = '/Users/damiano/Desktop/Master/Data_mining/Assignment_4/data/diabetes_train.csv'
    train = pd.read_csv(train_file)
    test_file = '/Users/damiano/Desktop/Master/Data_mining/Assignment_4/data/diabetes_test.csv'
    test = pd.read_csv(test_file)

    # extract first 7 columns to data matrix X
    X = train.iloc[:, 0:7].values
    X1 = test.iloc[:, 0:7].values
    # extract 8th column (labels) to numpy array
    Y = train.iloc[:, 7].values
    Y1 = test.iloc[:, 7].values

    scaler = StandardScaler()
    stand_train = scaler.fit_transform(X)
    stand_test = scaler.fit_transform(X1)
    model = LogisticRegression()
    model.fit(stand_train, Y)

    y_pred = model.predict(stand_test)

    print(compute_metrics(Y1, y_pred))

    ###################################################################

    print('Exercise 1.b')
    print('------------')
    print('I would choose Logistic Regression because it gives a higher accuracy in this data set\n')

    print('Exercise 1.c')
    print('------------')
    print('For another data set I would choose Logistic Regression because LDA can be used only on data with a normal distribution,'
          'depending on whether this condition is fulfilled LDA can be used as well\n')

    print('Exercise 1.d')
    print('------------')
    print('The two attributes that contribute most to the prediction are glu and ped.\n')
    coef = model.coef_
    exp_coef = np.exp(coef[0, 0])
    inc = 1 / (exp_coef + 1) * 100
    print('The coefficent for npreg is', round(coef[0, 0], 2), '. Calculating the exponential function results in', round(exp_coef, 2),
          'which amounts to an increase in diabetes risk of', round(inc, 2), '% per additional pregnancy.')
