#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

if __name__ == '__main__':

    iris = load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))

    ####################################################################

    def split_data(X, y, attribute_index, theta):
        X = pd.DataFrame(X)
        d1 = X[X.ix[:, attribute_index] < theta]
        d2 = X[X.ix[:, attribute_index] >= theta]
        return d1, d2

    def compute_information_content(d, y):
        X = pd.DataFrame(d)
        p = np.unique(y[X.index], return_counts=True)
        p = p[1]/len(d)
        info = - sum(p[i]*np.log2(p[i]) for i in range(len(p)))
        return info

    def compute_information_a(X, y, attribute_index, theta):
        d1, d2 = split_data(X, y, attribute_index, theta)
        infoA = len(d1)/len(X) * compute_information_content(d1, y) + len(d2)/len(X) * compute_information_content(d2, y)
        return infoA

    def compute_information_gain(x, y, attribute_index, theta):
        information_gain = compute_information_content(X, y) - compute_information_a(X, y, attribute_index, theta)
        return information_gain

    ####################################################################

    print('Exercise 2.b')
    print('------------')

    print('Split (', iris.feature_names[0], '< 5.5 ): information gain =', compute_information_gain(X, y, 0, 5.5))
    print('Split (', iris.feature_names[1], '< 3.0 ): information gain =', compute_information_gain(X, y, 1, 3.0))
    print('Split (', iris.feature_names[2], '< 2.0 ): information gain =', compute_information_gain(X, y, 2, 2.0))
    print('Split (', iris.feature_names[3], '< 1.0 ): information gain =', compute_information_gain(X, y, 3, 1.0))

    print('\nExercise 2.c')
    print('------------')

    print('When constructing a Decision Tree we want to progressively select attributes and thresholds that maximize the difference in cost \n'
          'between the split and non split data, this means that the computed value of the function information_gain has to be maximized. \n'
          'I would therefore choose either petal length (cm) < 2.0 or petal width (cm) < 1.0')

    ####################################################################
    # Exercise 2.d

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...

    print('\nExercise 2.d')
    print('------------')

    np.random.seed(42)

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        clf = DecisionTreeClassifier()
        clf = clf.fit(X[train_index], y[train_index])
        acc = accuracy_score(y[test_index], clf.predict(X[test_index]))
        feature_imp = clf.feature_importances_

    print('Accuracy score using cross-validation')
    print(round(100 * np.mean(acc), 2), '%\n')

    print('Feature importances for _original_ data set')
    print('-------------------------------------------')

    print(np.round_(feature_imp, 3))
    print('The two most important features are', iris.feature_names[2], 'and', iris.feature_names[3], '\n')

    X = X[y != 2]
    y = y[y != 2]

    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        clf = DecisionTreeClassifier()
        clf = clf.fit(X[train_index], y[train_index])
        acc = accuracy_score(y[test_index], clf.predict(X[test_index]))
        feature_imp = clf.feature_importances_

    print('Feature importances for _reduced_ data set')
    print('------------------------------------------')
    print(feature_imp)
    print('The most important feature for the reduced data set is',iris.feature_names[2])
    print('A value of 1 for feature importance means that we could classify the entire data set only splitting with this attribute')
