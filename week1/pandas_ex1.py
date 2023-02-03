"""
Exercises week 1 - DAT200
Repition of common pandas functions.
"""
import pandas as pd


# Getting data as a DataFrame
data = pd.read_csv('https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/iris/iris.data', header=None)

# Change col and row names
data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'types']
data.index = ['flower'+str(i) for i in range(1,151)]

# Find the unique values in types column
unique_types = data['types'].unique()

# Compute mean against types
mean_types = data.groupby('types').mean()

# Create new col
data['sepal width >=3'] = [True if sep_w >= 3 else False for sep_w in data['sepal width']]

# Count number of sepal width instances over 3
counts = data.groupby('sepal width >=3').size()

# Seperate the dataframe by types.
virginica = data[data['types'] == 'Iris-virginica']
setosa = data[data['types'] == 'Iris-setosa']
versicolor = data[data['types'] == 'Iris-versicolor']

# Count how many of each type
print(f'Number of: Virginica = {len(virginica)}, Setosa = {len(setosa)}, Versicolor = {len(versicolor)}')





