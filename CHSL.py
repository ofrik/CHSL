__author__ = 'Ofri'

from sklearn.base import clone
import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import numpy as np


class CHSL(object):
    def __init__(self, linearClassifier, clusteringAlgorithm, numberOfClusters):
        self.models = []
        self.baseLinearClassifier = clone(linearClassifier)
        self.numberOfClusters = numberOfClusters
        self.baseClusteringAlgorithm = clone(clusteringAlgorithm.set_params(**{"n_clusters": numberOfClusters}))

    def fit(self, X, y):
        self.models = []
        mjr_X, mjr_y, mnr_X, mnr_y, minorLabel = self.getMinorAndMajorDatasets(X, y)
        models = self.createSubDatasetsAndClassifiers(mjr_X, mjr_y, mnr_X, mnr_y)
        print models
        pass

    def createSubDatasetsAndClassifiers(self, mjr_X, mjr_y, mnr_X, mnr_y):
        """
        create and fit the classifiers with default parameters on the new sub datasets
        :param mjr_X: the major class train data
        :param mjr_y: the major class train labels
        :param mnr_X: the minor class train data
        :param mnr_y: the minor class train labels
        :return: list of the ensemble classifiers
        """
        clustersLabels = self.baseClusteringAlgorithm.fit(mjr_X).labels_
        for i in range(1, self.numberOfClusters + 1):
            tmp_clf = clone(self.baseLinearClassifier)
            subMjrClusterIndexes = np.where(clustersLabels == (i - 1))
            subMjrCluster_X = mjr_X.iloc[subMjrClusterIndexes]
            subMjrCluster_y = mjr_y.iloc[subMjrClusterIndexes]
            sub_X = pd.concat([subMjrCluster_X, mnr_X])
            sub_y = pd.concat([subMjrCluster_y, mnr_y])
            tmp_clf.fit(sub_X, sub_y)
            self.models.append(tmp_clf)
        return self.models

    def getMinorAndMajorDatasets(self, X, y):
        """
        split the dataset to major and minor datasets
        :param X: DataFame with the train data
        :param y: Series with the matching train labels
        :return: tuple of the major train data, major train labels, minor train data, minor train labels and the label of the minor class
        """
        unique = y.unique()
        total = y.count()
        indexes = np.where(y == unique[0])[0]
        if len(indexes) > total / 2:
            minorLabel = unique[1]
            majorIndexes = indexes
            minorIndexes = list(set([x for x in range(total)]) - set(indexes))
        else:
            minorLabel = unique[0]
            majorIndexes = list(set([x for x in range(total)]) - set(indexes))
            minorIndexes = indexes

        mjr_X = X.iloc[majorIndexes]
        mjr_y = np.take(y, majorIndexes)
        mnr_X = X.iloc[minorIndexes]
        mnr_y = np.take(y, minorIndexes)
        return mjr_X, mjr_y, mnr_X, mnr_y, minorLabel

    def predict(self, X):
        preds = []
        # predicting all the given samples
        for x in X:
            test_res = 1
            # check the results of the classifiers until one return the majority label
            for model in self.models:
                tmp_res = model.predict(x)
                if tmp_res == 0:
                    test_res = 0
                    break
            preds.append(test_res)
        return preds


if __name__ == "__main__":
    numberOfMajor = 1000
    numberOfMinor = 10
    mjr1 = [random.uniform(1, 100) for i in range(numberOfMajor)]
    mnr1 = [random.uniform(101, 200) for i in range(numberOfMinor)]
    mjr2 = [random.uniform(1, 50) for i in range(numberOfMajor)]
    mnr2 = [random.uniform(101, 150) for i in range(numberOfMinor)]
    df = pd.DataFrame(
        {"value1": mjr1 + mnr1, "value2": mjr2 + mnr2, "target": ([0] * numberOfMajor) + ([1] * numberOfMinor)})
    clf = CHSL(SVC(), KMeans(), 2)
    y = df["target"]
    X = df.drop("target", axis=1)
    clf.fit(X, y)
    print df
    print "started"
