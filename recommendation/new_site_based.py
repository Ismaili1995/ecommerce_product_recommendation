from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt

product_descriptions = pd.read_csv('dataset/data_storage/product_descriptions.csv')
product_descriptions.shape

# Missing values

product_descriptions = product_descriptions.dropna()
print(product_descriptions.shape)
print(product_descriptions.head())

product_descriptions1 = product_descriptions.head(500)
# product_descriptions1.iloc[:,1]

print(product_descriptions1["product_description"].head(10))

vectorizer = TfidfVectorizer(stop_words='english')
fiting_data = vectorizer.fit_transform(product_descriptions1["product_description"])
print(fiting_data)

# Fitting K-Means to the dataset

data = fiting_data

kmeans = KMeans(n_clusters = 10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(data)
plt.plot(y_kmeans, ".")
plt.show()

def print_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),

    

#Top words in each cluster based on product description
# # Optimal clusters is 

true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(data)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print_cluster(i)
    

#Predicting clusters based on key search words
def show_recommendations(product):
    #print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    print_cluster(prediction[0])

show_recommendations("cutting tool")