import pandas as pd

# Getting data as a DataFrame
data = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)

# Change col and row names
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'types']
data.index = ['flower'+str(i) for i in range(1,151)]



print(data)
