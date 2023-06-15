from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

cancer = load_breast_cancer()
acc_list = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, stratify=cancer.target, random_state=i)

    # Initialise standard scaler and compute mean and STD from training data
    sc = StandardScaler()
    sc.fit(X_train)

    # Transform (standardise) both X_train and X_test with mean and STD from
    # training data
    X_train_sc = sc.transform(X_train)
    X_test_sc = sc.transform(X_test)
    for c in np.arange(-8, 2):
        logreg = LogisticRegression(C=10.**c, penalty='l2',
                                    random_state=1).fit(X_train_sc, y_train)
        y_pred = logreg.predict(X_test_sc)
        acc_list.append(logreg.score(X_test_sc, y_test))

print(sum(acc_list)/len(acc_list))
