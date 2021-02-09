import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %matplotlib inline
plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD

data_ratings = pd.read_csv('dataset/data_storage/ratings_Beauty.csv')
data_ratings = data_ratings.dropna()
print(data_ratings.head())

print(data_ratings.shape)

popular_products = pd.DataFrame(data_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
print(most_popular.head(10))

plt.plot(most_popular.head(10))
plt.show()