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
print(data)
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

# View 10 last rows of sepal length and type
last_rows = data[['sepal length', 'types']]
last_rows = last_rows.iloc[-10:]

# View rows where sepal length > 5 and petal width < 0.2
cond_rows = data[(data['sepal length'] > 5) & (data['petal width'] < 0.2)]


# Make a new dataframe containing only the rows where petal width = 1.8
new_df = data[data['petal width'] == 1.8]


# Get descriptive stats for the whole dataframe and afterward for col petal length
stats_data = data.describe()

stats_petal_len = data['petal length'].describe()


# Remove rows named flower 55 and flower 77
data.drop(['flower55', 'flower77'], inplace=True)

# Remove col sepal width >= 3
data.drop('sepal width >=3', axis=1, inplace=True)

# View all rows of sepal length where petal width is exactly 1.8
spec = data['sepal length'].where(data['petal width'] == 1.8)
spec.dropna(inplace=True)

# Get vals of df stored in np array (in practice remove cols and rows)
df = data[:].values

# Remove col types and apply func name computation to each cell in data. Func should do
# the following: take val of cell, add 1 and multiply by 3.
data.drop('types', axis=1, inplace=True)
data = data.applymap(lambda x: (x+1)*3)


