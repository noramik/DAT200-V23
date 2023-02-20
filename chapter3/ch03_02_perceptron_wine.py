from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

wine = datasets.load_wine()

X = wine.data[:, [0, 10]]
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5, stratify=y)

ppn = Perceptron(max_iter=100, eta=0.05, random_state=77)


