import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

dataset = pd.read_csv('dataset/lyrics_train.csv', engine='python')
feature = dataset['lyrics']
target = dataset['genre']

# TODO - Choose 'test_size' and 'random_state'
X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.9, random_state=0)

# TODO - Make your own features
# Example : TF-IDF vector
X_train_feature = TfidfTransformer().fit_transform(CountVectorizer(max_features=100).fit_transform(X_train)).toarray()
X_test_feature = TfidfTransformer().fit_transform(CountVectorizer(max_features=100).fit_transform(X_test)).toarray()

# TODO - 2-1-1. Build pipeline for Naive Bayes Classifier
clf_nb = Pipeline([])
clf_nb.fit(X_train_feature, Y_train)

# TODO - 2-1-2. Build pipeline for SVM Classifier
clf_svm = Pipeline([])
clf_svm.fit(X_train_feature, Y_train)

predicted = clf_nb.predict(X_test_feature)
# predicted = clf_svm.predict(X_test_feature)

print("accuracy : %d / %d" % (np.sum(predicted==Y_test), len(Y_test)))

with open('model_nb.pkl', 'wb') as f1:
    pickle.dump(clf_nb, f1)

with open('model_svm.pkl', 'wb') as f2:
    pickle.dump(clf_svm, f2)