import random
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import pandas as pd
from CHSL import CHSL
import re
import os


def getDataset():
    baseDir = "datasets"
    regex = r"@inputs\s(?P<columns>.*)"
    for path in os.listdir(baseDir):
        dataFile = "%s\\%s\\%s.dat" % (baseDir, path, path)
        names = []
        with open(dataFile, "r+") as f:
            line = f.readline()
            stop = False
            while not stop:
                matches = re.search(regex, line)
                if matches is not None:
                    stop = True
                    names = matches.group("columns").split(", ")
                    names.append("target")
                else:
                    line = f.readline()
        yield pd.read_csv(dataFile, names=names, comment='@')


if __name__ == "__main__":
    for df in getDataset():
        y = df["target"]
        X = df.drop("target", axis=1)
        clf = SVC(kernel="linear")
        clf.fit(X,y)
        print clf


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
