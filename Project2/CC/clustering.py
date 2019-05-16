import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.cluster import KMeans

dataset = pd.read_csv('dataset/lyrics_train.csv', engine='python')
feature = dataset['lyrics']
target = dataset['genre']

# TODO - Data preprocessing and clustering
# TODO - Set Randomness parameters to specific value(ex: random_state in KMeans etc.) Or Save KMeans model to pickle file
data_trans = TfidfTransformer().fit_transform(CountVectorizer().fit_transform(feature))
clst = KMeans(n_clusters=7)
clst.fit(data_trans)

print(metrics.v_measure_score(target, clst.labels_))