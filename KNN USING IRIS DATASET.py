# K NEAREST NEIGHBOUR USING IRIS DATASET

# LOADING REQUIRED MODULES
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# LOADING DATASET
iris = datasets.load_iris()

# PRINTING DESCRIPTION AND FEATURES
# print(iris.DESCR)
features = iris.data
labels = iris.target
print(features[0], labels[0])

# TRAINING THE CLASSIFIER
clf = KNeighborsClassifier()
clf.fit(features, labels)

preds = clf.predict([[1, 1, 1, 1]])
print(preds)
