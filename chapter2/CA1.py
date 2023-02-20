#!/usr/bin/env python
# coding: utf-8

# # Compulsory Assignment 1 - Pandas and visualizations

# ### Nora Mikarlsen 

# ### Imports

# In[3]:
                åp0''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ---
# ## Loading and exploring the dataset

# __1. Load the dataset named `airbnb.csv` and store it in a dataframe called `raw_df`. Use the column named `ìd` as the index column for the dataframe__

# In[4]:


# Insert your code below
# ======================
raw_df = pd.read_csv("CA1/assets/airbnb.csv", index_col=0)



# __2. Print the first `five` rows of the dataframe__

# In[5]:


# Insert your code below
# ======================
# pandas head method prints the first five rows of the dataframe by default. 
raw_df.head()


# __3. How many unique values exist in each of the columns `state` and `city`?__

# In[6]:


# Insert your code below
# ======================
unique_states = raw_df['state'].unique()
unique_cities = raw_df['city'].unique()
print(f'Number of unique values for states: {len(unique_states)} \nNumber of unique values for cities: {len(unique_cities)}')


# __4. Identify missing (NaN) values in each of the columns in the dataset__

# In[21]:


# Insert your code below
# ======================
print(f'Number of missing values per column:')
for col in raw_df.columns:
    print(f'{col}: {raw_df[col].isnull().sum()}')


# __5. Create a copy of `raw_df` named `df`. Remove any rows containing NaN values in the new dataframe. What is the shape of `df` before and after removing the NaN values?__

# In[22]:


# Insert your code below
# ======================
df = raw_df.dropna()
print(f'Shape of dataframe before removing NaN: {raw_df.shape}\nShape of dataframe after removing NaN: {df.shape}')
print(f'{len(raw_df.index)-len(df.index)} rows have been dropped from the dataframe because they contained NaN values.')


# __6. Which `room_type`, `state` and `city` is the most popular (by number of instances)? Print the name and count of each__
# 
# Hint: The output should look something like this:
# ```python
# Column: [col], Most popular: [name], Count: [count]
# Column: [col], Most popular: [name], Count: [count]
# Column: [col], Most popular: [name], Count: [count]
# ```

# In[25]:


# Insert your code below
# ======================

room_counts = df['room_type'].value_counts()
print(f'Column: room_type, Most popular: {room_counts.index[0]}, Count: {max(room_counts)}')

state_counts = df['state'].value_counts()
print(f'Column: state, Most popular: {state_counts.index[0]}, Count: {max(state_counts)}')

city_counts = df['city'].value_counts()
print(f'Column: city, Most popular: {city_counts.index[0]}, Count: {max(city_counts)}')


# __7. What is the average and median `price` for a listing?__

# In[26]:


# Insert your code below
# ======================
print(f"The mean price for a listing is: {df['price'].mean()} \nThe median price for a listing is: {df['price'].median()}")


# __8. What is the average price for the states `CA`, `FL` and `NY`?__
# 
# Hint: The output should look something like this:
# ```python
# State: [col], Average price: [price]
# State: [col], Average price: [price]
# State: [col], Average price: [price]
# ```

# In[13]:


# Insert your code below
# ======================
mean_prices = df.groupby(['state'])['price'].mean()
print(f"State: CA, Average price: {mean_prices['CA']}")
print(f"State: FL, Average price: {mean_prices['FL']}")
print(f"State: NY, Average price: {mean_prices['NY']}")


# __9. Create a new dataframe called `df_beach` containing all listings with "beach" in the `name`. Print out the shape of `beach_df`__
# 
# The filtering should not be case sensitive, meaning that names containing `beach`, `Beach`, `BeAcH` etc. all should be included
# 

# In[27]:


# Insert your code below
# ======================
df_beach = df.query('name.str.contains("beach", case=False)')
print(f'The shape of the df_beach dataframe is {df_beach.shape}')
print(f'{round(df_beach.shape[0]/df.shape[0]*100, 3)} percent of the airbnbs contained the word beach in the title')


# ---
# 
# ## Visualizing the dataset

# __10. Create plot with 2 vertical axes and one horizontal axes. The plot should display a barchart containing the `count` of the `10 most popular` states and cities, each in its own subplot. The bars should be sorted in descending order.__
# 
# Use `df` in all tasks in this section
# 
# Hint: It is recommended to use the `Barplot` function built into Seaborn for barcharts.
# 
# The output should look something like this:
# 
# <img src="assets/ex10.png"
#      alt="Barchart example"/>
# 
# PS: Disregard the color scheme of the example image.

# In[15]:


# Insert your code below
# ======================

plt.figure()
sns.barplot(data=df, x=df.state.value_counts().index[0:10], y=df.state.value_counts()[0:10])
plt.figure()
ax = sns.barplot(data=df, x=df.city.value_counts().index[0:10], y=df.city.value_counts()[0:10] )
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()


# __11. Create a scatterplot with the longitude and latitude of the listings in `df`. Longitude should be on the x-axis and latitude on the y-axis.__ 
# 
# The output should look something like this:
# 
# <img src="assets/ex11.png"
#      alt="Scatterplot example"/>
# 
# PS: Disregard the color scheme of the example image.

# In[16]:


# Insert your code below
# ======================
sns.scatterplot(data=df, x=df['longitude'], y=df['latitude'])


# __12. Create a matrix containing the correlations between the different columns in `df`. Plot it as a heatmap using Seaborn or similar. What does the plot tell you about correlations? Which columns are the most correlated to `price`?__

# In[20]:


# Insert your code below
# ======================
correlations = df.corr()
plt.figure()
sns.heatmap(correlations, vmin=-1, vmax=1)


# The correlation matrix shows how strongly two columns in the dataframe are correlated. The heatbar shows what the different colours mean for correlation, white means perfectly correlated, while black means perfectly negatively correlated, and pure red is no correlation at all. 
# 
# From the correlation matrix we see that all the boxes on the diagonal are white, meaning perfect correlation, this makes sense as this basically says all the columns are correlated with themselves. 
# 
# The price seems to be most strongly correlated with calculated_host_listings_count, as that is the box with the lightest colour in that row. There also seems to be some slight negative correlation between price and latitude and longitude. 
