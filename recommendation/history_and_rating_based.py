import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD

data_ratings = pd.read_csv('https://drive.google.com/file/d/14AFyhvXz1CwjL_gwMCA2kUJ9zLv8obqC/view?usp=sharing')
data_ratings = data_ratings.dropna()
data_ratings.head()

data_ratings1 = data_ratings.head(10000)

ratings_utility_matrix = data_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
print(ratings_utility_matrix.head())
print(ratings_utility_matrix.shape)

X = ratings_utility_matrix.T
print(X.head())
print(X.shape)

X1 = X

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
print(decomposed_matrix.shape)

correlation_matrix = np.corrcoef(decomposed_matrix)
print(correlation_matrix.shape)

X.index[99]

i = "6117036094"

product_names = list(X.index)
product_ID = product_names.index(i)
print(product_ID)

correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape

Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i) 

print(Recommend[0:9])