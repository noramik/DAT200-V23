# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:52:10 2018

@author: olto
"""

# ==============================================================================
# Import modules
# ==============================================================================

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
# ==============================================================================
# Load data
# ==============================================================================

bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# bc = datasets.load_wine()
# X = bc.data
# y = bc.target


# ==============================================================================
# Run loops across values of C and for many train test splits
# ==============================================================================

# Run loop across increasing values of C and collect accuracies. Each accuracy
# will be based on many train test splits
allTrainAcc = []
allTestAcc = []

c_range = np.arange(-8, 5)
for c in c_range:

    # Run loop across increasing values for random_state of train_test_split
    # and collect for each split
    accTrainList = []
    accTestList = []

    for rs in range(1, 101):
        # Split data into training and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=rs, stratify=y)

        # ==============================================================================
        # Scale features using StandardScaler class in scikit-learn
        # ==============================================================================

        # Initialise standard scaler and compute mean and STD from training data
        sc = StandardScaler()
        sc.fit(X_train)

        # Transform (standardise) both X_train and X_test with mean and STD from
        # training data
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        # ==============================================================================
        # Train multiclass logistic regression classifier from in scikit-learn
        # ==============================================================================

        # Initialise the model
        logR = LogisticRegression(C=10. ** c,
                                  random_state=1,
                                  solver='liblinear',
                                  multi_class='auto')

        logR.fit(X_train_sc, y_train)

        # ==============================================================================
        # Compute performance metrics
        # ==============================================================================

        # Compute accuracies for train and test data for 1 train test split
        accTrainList.append(logR.score(X_train_sc, y_train))
        accTestList.append(logR.score(X_test_sc, y_test))

    # Compute average train accuracy and STD across all train test splits and
    # print out results
    accTrain_aver = np.mean(accTrainList)
    accTrain_std = np.std(accTrainList)
    print('Av. training accuracy for C=10**{0:.4f} is {1:.4f} +/- {2:.3f}'.format(c, accTrain_aver,
                                                                                  accTrain_std))

    # Compute average test accuracy and STD across all train test splits and
    # print out results
    accTest_aver = np.mean(accTestList)
    accTest_std = np.std(accTestList)
    print('Av. test accuracy for C=10**{0:.4f} is {1:.4f} +/- {2:.3f}'.format(c, accTest_aver,
                                                                              accTest_std))
    print('\n')

    # Collect average accuracy for each C
    allTrainAcc.append(accTrain_aver)
    allTestAcc.append(accTest_aver)

# ==============================================================================
# Plot train and test accuracies across regularisation paramter C
# ==============================================================================

# Construct pandas dataframe from lists with accuracies
accuracies = {'train acc': allTrainAcc, 'test acc': allTestAcc}
acc_df = pd.DataFrame(data=accuracies)

# Add column holding strings indicating value of C. Needed for xticks lables
acc_df['C'] = ['10**{0}'.format(c) for c in c_range]

# Plot columns acc train and acc test. Define xticks lables
plt.figure()
ax = acc_df.plot(xticks=acc_df.index, rot=45)
ax.set_xticklabels(acc_df.C)
plt.show()