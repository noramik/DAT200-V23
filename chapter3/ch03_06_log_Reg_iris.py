#==============================================================================
# Import modules
#==============================================================================

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt



#==============================================================================
# Load data and select features
#==============================================================================
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target



#==============================================================================
# Split into training and test data
#==============================================================================
# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)



#==============================================================================
# Scale features using StandardScaler class in scikit-learn
#==============================================================================
# Initialise standard scaler and compute mean and STD from training data
sc = StandardScaler()
sc.fit(X_train)


# Transform (standardise) both X_train and X_test with mean and STD from
# training data
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)



#==============================================================================
# Train logistic regression model from scikit-learn
#==============================================================================
lr = LogisticRegression(C=100.0,
                        random_state=1,
                        solver='liblinear',
                        multi_class='auto')
lr.fit(X_train_sc, y_train)



#==============================================================================
# Plot the data
#==============================================================================
X_combined_sc = np.vstack((X_train_sc, X_test_sc))
y_combined = np.hstack((y_train, y_test))


# Specify keyword arguments to be passed to underlying plotting functions
scatter_kwargs = {'s': 120, 'edgecolor': None, 'alpha': 0.7}
contourf_kwargs = {'alpha': 0.2}
scatter_highlight_kwargs = {'s': 120, 'label': 'Test data', 'alpha': 0.7}


# Plotting decision regions
plt.figure(figsize=(8,6))
plot_decision_regions(X=X_combined_sc,
                      y=y_combined,
                      clf=lr,
                      X_highlight=X_test_sc,
                      scatter_kwargs=scatter_kwargs,
                      contourf_kwargs=contourf_kwargs,
                      scatter_highlight_kwargs=scatter_highlight_kwargs)


plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_06.png', dpi=300)
plt.show()



#==============================================================================
# Print out accuracy
#==============================================================================
print('Accuracy: {0:.2f}'.format(lr.score(X_test_sc, y_test)))



#==============================================================================
# Compute class probabilities and predicted class
#==============================================================================

# Class probabilities for the test set
y_test_prob = lr.predict_proba(X_test_sc)


# Predict classes for test set
y_test_pred = lr.predict(X_test_sc)


# Predict a single flower sample
y_test_pred_oneFlower = lr.predict(X_test_sc[0, :].reshape(1, -1))






