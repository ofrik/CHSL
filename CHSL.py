__author__ = 'Ofri'

from sklearn.base import clone
import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
import time
import warnings

warnings.filterwarnings("ignore")

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap


class CHSL(object):
    def __init__(self, linearClassifier, clusteringAlgorithm, numberOfClusters, paramGrid, b=0.7, n_jobs=1):
        """
        Initialize the CHSL model with the linear classifier that will be created for each dataset that will be created
        from the original dataset, the clustering algorithm to create the majority sub-datasets, the number of
        sub-datasets to split to, the parameters space to optimize, the minimum specificity threshold and the number of
        jobs to run in parallel in the grid search
        :param linearClassifier: The base linear classifier that will be used
        :param clusteringAlgorithm: The clustering algorithm to be used
        :param numberOfClusters: The number of clusters to create from the majority dataset
        :param paramGrid: The parameters space for optimization
        :param b: The minimum threshold for the specificity
        :param n_jobs: The number of jobs for the grid search
        """
        self.models = []
        self.baseLinearClassifier = clone(linearClassifier)
        self.numberOfClusters = numberOfClusters
        self.baseClusteringAlgorithm = clone(clusteringAlgorithm.set_params(**{"n_clusters": numberOfClusters}))
        self.paramGrid = paramGrid
        self.scoreFunc = make_scorer(self.score_func, b=b)
        self.n_jobs = n_jobs

    @timing
    def fit(self, X, y):
        """
        Fit the CHSL model by splitting the majority class data into p sub-datasets and creating a model for each
        majority sub-dataset with the minority dataset.
        After these are fitted using grid search to find the best parameters in the given range and to train the models
        accordingly
        :param X: Train data
        :param y: Train labels
        :return: Self
        """
        self.models = []
        mjr_X, mjr_y, mnr_X, mnr_y, minorLabel = self.getMinorAndMajorDatasets(X, y)
        models = self.createSubDatasetsAndClassifiers(mjr_X, mjr_y, mnr_X, mnr_y)
        modelOptimizer = CHSLOptimizer(models)
        clf = GridSearchCV(modelOptimizer, param_grid=self.paramGrid, scoring=self.scoreFunc, cv=10, verbose=1,
                           refit=False, n_jobs=self.n_jobs)
        X, y = shuffle(X, y, random_state=0)
        clf.fit(X, y)
        bestParams = clf.best_params_
        modelOptimizer.set_params(bestParams)
        modelOptimizer.fit(X, y)
        return self

    def createSubDatasetsAndClassifiers(self, mjr_X, mjr_y, mnr_X, mnr_y):
        """
        Create and fit the classifiers with default parameters on the new sub datasets
        :param mjr_X: The major class train data
        :param mjr_y: The major class train labels
        :param mnr_X: The minor class train data
        :param mnr_y: The minor class train labels
        :return: List of the ensemble classifiers
        """
        clustersLabels = self.baseClusteringAlgorithm.fit(mjr_X).labels_
        for i in range(0, self.numberOfClusters):
            mnr_y[mnr_y != 1] = 1
            tmp_clf = clone(self.baseLinearClassifier)
            subMjrClusterIndexes = np.where(clustersLabels == i)
            subMjrCluster_X = mjr_X.iloc[subMjrClusterIndexes]
            subMjrCluster_y = mjr_y.iloc[subMjrClusterIndexes]
            subMjrCluster_y[subMjrCluster_y != 0] = 0
            sub_X = pd.concat([subMjrCluster_X, mnr_X])
            sub_y = pd.concat([subMjrCluster_y, mnr_y])
            tmp_clf.fit(sub_X, sub_y)
            self.models.append(tmp_clf)
        return self.models

    def getMinorAndMajorDatasets(self, X, y):
        """
        Split the dataset to major and minor datasets
        :param X: DataFame with the train data
        :param y: Series with the matching train labels
        :return: Tuple of the major train data, major train labels, minor train data, minor train labels and the label of the minor class
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
        """
        Predict the class of the given data by iterating over the models created in the fit stage. an instance is
        determined to be the minority class only if none of the models predicted the instance to be a majority class
        :param X: Test data
        :return: List of predictions
        """
        preds = []
        # predicting all the given samples
        for i, x in X.iterrows():
            test_res = 1
            # check the results of the classifiers until one return the majority label
            for model in self.models:
                tmp_res = model.predict(x)[0]
                if tmp_res == 0:
                    test_res = 0
                    break
            preds.append(test_res)
        return preds

    def score_func(self, y, y_pred, b):
        """
        Score for the CV, positive is labeling 1 (the minority class)
        :param y: Actual labels
        :param y_pred: Predicted labels
        :param b: The specificity threshold
        :return: The sensitivity of the predictions if satisfy the minimum specificity otherwise 0 (as the lowest
        possible sensitivity)
        """
        matrix = confusion_matrix(y, y_pred)
        FP = (matrix.sum(axis=0) - np.diag(matrix))[0]
        FN = (matrix.sum(axis=1) - np.diag(matrix))[0]
        TP = np.diag(matrix)[0]
        TN = np.sum(matrix) - (FP + FN + TP)
        try:
            sensitivity = float(TP) / (TP + FN)
        except ZeroDivisionError as inst:
            sensitivity = 1
        try:
            specificity = float(TN) / (TN + FP)
        except ZeroDivisionError as inst:
            specificity = 1
        if specificity > b:
            return sensitivity
        else:
            return 0


class CHSLOptimizer(CHSL, BaseEstimator):
    """
    This class is a wrapper for the CHSL models to allow easy optimization of the models parameters using grid search
    method of sklearn in order to learn the the best configuration for the model
    """

    def __init__(self, models):
        """
        Initialize the class with the models to be optimized
        :param models: List of linear models
        """
        self.models = models

    def fit(self, X, y=None):
        """
        Fit the models with the given X, y
        :param X: Train data
        :param y: Train labels
        :return: Self
        """
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        Using the parent method to predict the class of all the given instances
        :param X: Test data
        :return: List of predictions of the CHSL model
        """
        return super(CHSLOptimizer, self).predict(X)

    def set_params(self, **params):
        """
        Set the given parameters for all the models
        :param params: Parameters to be set
        :return: Self
        """
        for model in self.models:
            model.set_params(**params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of this instance
        :param deep: Not being used, only for signature compatibility
        :return: Dictionary of the model parameters and the values
        """
        return {"models": self.models}


if __name__ == "__main__":
    numberOfMajor = 1000
    numberOfMinor = 10
    mjr1 = [random.uniform(1, 100) for i in range(numberOfMajor)]
    mnr1 = [random.uniform(101, 200) for i in range(numberOfMinor)]
    mjr2 = [random.uniform(1, 50) for i in range(numberOfMajor)]
    mnr2 = [random.uniform(101, 150) for i in range(numberOfMinor)]
    df = pd.DataFrame(
        {"value1": mjr1 + mnr1, "value2": mjr2 + mnr2, "target": ([0] * numberOfMajor) + ([1] * numberOfMinor)})
    clf = CHSL(SVC(kernel="linear"), KMeans(), 3, {"C": [1, 2]})
    y = df["target"]
    X = df.drop("target", axis=1)
    clf.fit(X, y)
    print "started"
