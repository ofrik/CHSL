import random
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.cluster import KMeans
import pandas as pd
from CHSL import CHSL
import re
import os
import time
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil
from sklearn.base import clone

RANDOM_SEED = 5


def getDataset():
    baseDir = "datasets"
    regex = r"@inputs?\s(?P<columns>.*)"
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
        yield pd.read_csv(dataFile, names=names, comment='@'), categoricalColumns, path


def trainData():
    for df, categoricalColumns, dataset_name in getDataset():
        # print df, categoricalColumns
        y = df["target"].apply(lambda x: 1 if x.strip() == "positive" else 0)
        for col in categoricalColumns:
            df[col] = pd.Categorical.from_array(df[col]).codes
        X = df.drop("target", axis=1)
        yield X, y, dataset_name


def calcSensitivityAndSpecificity(y_actual, y_pred):
    matrix = confusion_matrix(y_actual, y_pred)
    FP = (matrix.sum(axis=0) - np.diag(matrix))[0]
    FN = (matrix.sum(axis=1) - np.diag(matrix))[0]
    TP = np.diag(matrix)[0]
    TN = np.sum(matrix) - (FP + FN + TP)
    return TP, TN, FP, FN


def checkClaasifier(dataset_name, classifier_name, fold, clf, X_train, y_train, X_test, y_test, numberOfClusters,
                    params={}):
    print "Working on %s using %s - %s" % (dataset_name, classifier_name, fold)
    clf.set_params(**params)
    time1 = time.time()
    clf.fit(X_train, y_train)
    time2 = time.time()
    # print "Training %s with %s took %0.3f ms" % (classifier_name, dataset_name, (time2 - time1) * 1000.0)
    time3 = time.time()
    y_pred = clf.predict(X_test)
    time4 = time.time()
    TP, TN, FP, FN = calcSensitivityAndSpecificity(y_test, y_pred)
    try:
        sensitivity = float(TP) / (TP + FN)
    except ZeroDivisionError as inst:
        sensitivity = 1
    try:
        specificity = float(TN) / (TN + FP)
    except ZeroDivisionError as inst:
        specificity = 1
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/%s_%s.csv" % (dataset_name, classifier_name)):
        with open("results/%s_%s.csv" % (dataset_name, classifier_name), 'w+') as f:
            f.write("Fold,Number_Of_Clusters,TP,TN,FP,FN,Senitivity,Specificity,Train Time,Prediction Time\n")
    with open("results/%s_%s.csv" % (dataset_name, classifier_name), 'a+') as f:
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
            fold, numberOfClusters, TP, TN, FP, FN, sensitivity, specificity, (time2 - time1) * 1000.0,
            (time4 - time3) * 1000.0))

    return sensitivity, specificity, (time2 - time1) * 1000.0, (time4 - time3) * 1000.0


if __name__ == "__main__":
    classifiers = [
        ("SVC", SVC(kernel="linear", max_iter=10000, random_state=RANDOM_SEED, class_weight='balanced'),
         {"C": np.arange(1.0, 10.0, 1.0)}),
        ("Perceptron", Perceptron(random_state=RANDOM_SEED, class_weight='balanced'),
         {"penalty": [None, "l2", "l1", "elasticnet"], "alpha": np.arange(0.0001, 0.01, 0.0015)}),
        ("PassiveAggressiveClassifier", PassiveAggressiveClassifier(random_state=RANDOM_SEED, class_weight='balanced'),
         {"C": np.arange(1.0, 10.0, 1.0)}),
        ("RidgeClassifier", RidgeClassifier(random_state=RANDOM_SEED, max_iter=10000, class_weight='balanced'),
         {"alpha": np.arange(0.0001, 5, 0.5)}),
        ("SGDClassifier", SGDClassifier(random_state=RANDOM_SEED, class_weight='balanced'),
         {"loss": ["hinge", "log", "modified_huber", "squared_hinge"]})
    ]
    if os.path.exists("results"):
        shutil.rmtree("results")
    try:
        for numberOfClusters in range(5, 20, 3):
            for classifier in classifiers:
                for X, y, dataset_name in trainData():
                    print "Train size: %s" % (len(X))
                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
                    clf = clone(classifier[1])
                    chsl = CHSL(clone(classifier[1]), KMeans(random_state=RANDOM_SEED),
                                numberOfClusters, classifier[2])
                    avgOriginalSensitivity = []
                    avgOriginalSpecificity = []
                    avgOriginalTrainTime = []
                    avgOriginalPredictTime = []
                    avgCHSLSensitivity = []
                    avgCHSLSpecificity = []
                    avgCHSLTrainTime = []
                    avgCHSLPredictTime = []
                    counter = 1
                    for train_index, test_index in skf.split(X, y):
                        # print("TRAIN:", train_index, "TEST:", test_index)
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                        sensitivity, specificity, train_time, predict_time = checkClaasifier(dataset_name,
                                                                                             "CHSL_with_%s" %
                                                                                             classifier[0],
                                                                                             counter, chsl, X_train,
                                                                                             y_train, X_test, y_test,
                                                                                             numberOfClusters)
                        avgCHSLSensitivity.append(sensitivity)
                        avgCHSLSpecificity.append(specificity)
                        avgCHSLTrainTime.append(train_time)
                        avgCHSLPredictTime.append(predict_time)

                        sensitivity, specificity, train_time, predict_time = checkClaasifier(dataset_name,
                                                                                             classifier[0],
                                                                                             counter, clf, X_train,
                                                                                             y_train,
                                                                                             X_test, y_test,
                                                                                             numberOfClusters)
                        avgOriginalSensitivity.append(sensitivity)
                        avgOriginalSpecificity.append(specificity)
                        avgOriginalTrainTime.append(train_time)
                        avgOriginalPredictTime.append(predict_time)
                        counter += 1
                    print "The average CHSL sensitivity %s, specificity %s, train time %s ms, prediction time: %s ms" % (
                        np.mean(avgCHSLSensitivity), np.mean(avgCHSLSpecificity), np.mean(avgCHSLTrainTime),
                        np.mean(avgCHSLPredictTime))
                    print "The average sensitivity %s, specificity %s, train time %s ms, prediction time: %s ms\n" % (
                        np.mean(avgOriginalSensitivity), np.mean(avgOriginalSpecificity), np.mean(avgOriginalTrainTime),
                        np.mean(avgOriginalPredictTime))
    except Exception as inst:
        print "ERROR:", inst
