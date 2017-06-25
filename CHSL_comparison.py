import random
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pandas as pd
from CHSL import CHSL
import re
import os
import time
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix


def getDataset():
    baseDir = "datasets"
    regex = r"@inputs\s(?P<columns>.*)"
    regex2 = r"@attribute\s(?P<column>.*)\s{(?P<values>.*)}"
    for path in os.listdir(baseDir):
        dataFile = "%s\\%s\\%s.dat" % (baseDir, path, path)
        names = []
        categoricalColumns = []
        with open(dataFile, "r+") as f:
            line = f.readline()
            stop = False
            while not stop:
                matches = re.search(regex, line)
                columnTypeAndValues = re.search(regex2, line)
                if columnTypeAndValues is not None:
                    column = columnTypeAndValues.group("column")
                    if column != "Class":
                        categoricalColumns.append(column)
                if matches is not None:
                    stop = True
                    names = matches.group("columns").split(", ")
                    names.append("target")
                else:
                    line = f.readline()
        print path
        yield pd.read_csv(dataFile, names=names, comment='@'), categoricalColumns


def trainData():
    for df, categoricalColumns in getDataset():
        # print df, categoricalColumns
        y = df["target"].apply(lambda x: 1 if x.strip() == "positive" else 0)
        for col in categoricalColumns:
            df[col] = pd.Categorical.from_array(df[col]).codes
        X = df.drop("target", axis=1)
        yield X, y


def calcSensitivityAndSpecificity(y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
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
    return sensitivity, specificity


def checkClaasifier(clf, X_train, y_train, X_test, y_test, params={}):
    clf.set_params(**params)
    time1 = time.time()
    clf.fit(X_train, y_train)
    time2 = time.time()
    print "Training took %0.3f ms" % ((time2 - time1) * 1000.0)
    y_pred = clf.predict(X_test)
    return calcSensitivityAndSpecificity(y_test, y_pred)


if __name__ == "__main__":
    for X, y in trainData():
        print "Train size: %s" % (len(X))
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        clf = SVC(kernel="linear",
                  max_iter=10000)
        chsl = CHSL(SVC(kernel="linear", max_iter=10000), KMeans(), 10, {"C": np.arange(1.0, 10.0, 1.0)})
        avgOriginalSensitivity = []
        avgOriginalSpecificity = []
        avgCHSLSensitivity = []
        avgCHSLSpecificity = []
        for train_index, test_index in skf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            sensitivity, specificity = checkClaasifier(chsl, X_train, y_train, X_test, y_test)
            avgCHSLSensitivity.append(sensitivity)
            avgCHSLSpecificity.append(specificity)

            sensitivity, specificity = checkClaasifier(clf, X_train, y_train, X_test, y_test)
            avgOriginalSensitivity.append(sensitivity)
            avgOriginalSpecificity.append(specificity)
        print "The average CHSL sensitivity %s, specificity %s" % (
            np.mean(avgCHSLSensitivity), np.mean(avgCHSLSpecificity))
        print "The average sensitivity %s, specificity %s\n" % (
            np.mean(avgOriginalSensitivity), np.mean(avgOriginalSpecificity))




        # numberOfMajor = 1000
        # numberOfMinor = 10
        # mjr1 = [random.uniform(1, 100) for i in range(numberOfMajor)]
        # mnr1 = [random.uniform(101, 200) for i in range(numberOfMinor)]
        # mjr2 = [random.uniform(1, 50) for i in range(numberOfMajor)]
        # mnr2 = [random.uniform(101, 150) for i in range(numberOfMinor)]
        # df = pd.DataFrame(
        #     {"value1": mjr1 + mnr1, "value2": mjr2 + mnr2, "target": ([0] * numberOfMajor) + ([1] * numberOfMinor)})
        # clf = CHSL(SVC(kernel="linear"), KMeans(), 3, {"C": [1, 2]})
        # y = df["target"]
        # X = df.drop("target", axis=1)
        # clf.fit(X, y)
        # print "started"
