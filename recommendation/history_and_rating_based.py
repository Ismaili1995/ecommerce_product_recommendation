import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD

amazon_ratings = pd.read_csv('../input/amazon-ratings/ratings_Beauty.csv')
amazon_ratings = amazon_ratings.dropna()
amazon_ratings.head()

amazon_ratings1 = amazon_ratings.head(10000)

ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
ratings_utility_matrix.head()
ratings_utility_matrix.shape

X = ratings_utility_matrix.T
X.head()
X.shape

X1 = X

SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape

correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape

X.index[99]

i = "6117036094"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID

correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape

Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i) 

Recommend[0:9]