import graphviz
import pandas as pd
from sklearn import tree
import numpy as np
import os

from graphviz import Source
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def Ridge(data_train, y_train, lamb):
    I = np.eye(data_train.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(data_train.T, data_train) + lamb * I), data_train.T), y_train)
    return w

# opening the original file and memorizing it as a string
f = open("chronic_kidney_disease.arff", 'r')
string = f.read()

# CLEANING THE DATASET
# removing all types of errors we don't want into the data [like double commas]
string = string.replace(',,', ',')
string = string.replace(',\n', '\n')
string = string.replace(',\t', ',')
string = string.replace('\t', '')
cond = True
while cond == True:
    if ', ' in string:
        string = string.replace(', ', ',')
    else:
        cond = False

# names of the features
featNames = ['age', 'blood pressure', 'specific gravity', 'albumin', 'sugar', 'red blood cells', 'pus cell',
             'pus cell clumps', 'bacteria', 'blood glucose random', 'blood urea', 'serum creatinine', 'sodium',
             'potassium', 'hemoglobin', 'packed cell volume', 'white blood cell count', 'red blood cell count',
             'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema', 'anemia',
             'class']
# opening an empty file and saving here the corrected data
f2 = open("kidney.arff", 'w')
f2.write(string)
# colising the opened file
f.close()
f2.close()
# we open the corrected file using it as data [setting ',' as serpator, no header, skiping inital rows and setting '?' as NaN, names of the columns/features]
df = pd.read_csv('kidney.arff', sep=',', skiprows=29, header=None, na_values=['?', '\t?'], names=featNames)
# print(x)

# REPLACING CATHEGORICAL FEATURES WITH NUMERICAL (binary) FEATURES
replacement = {'normal': 0, 'abnormal': 1,
               'present': 0, 'notpresent': 1,
               'good': 0, 'poor': 1,
               'yes': 0, '\tyes': 0, ' yes': 0, 'no': 1, #All type of Yes
               'notckd': 0, 'ckd': 1}
df = df.replace(to_replace=replacement)


# MANAGEMENT OF MISSING DATA
X_to_cor = df.dropna(thresh=20).values
X = df.dropna(thresh=25).values.astype(float)

# normalizing X
st_dv = X.std(axis=0)
mean = X.mean(axis=0)
X_norm = (X - mean) / st_dv
rows, columns = X_to_cor.shape
data_train = np.copy(X_norm)
xcor = X_to_cor.copy()

for i in range(rows):
    row = X_to_cor[i, :]
    c = 0
    c_index = np.zeros((5, 1), dtype=int)
    for j in range(columns):
        if np.isnan(row[j]):
            c_index[c] = j
            c += 1
    if c != 0:
        index = np.zeros((c, 1), dtype=int)
        for k in range(c):
            index[k] = c_index[k]

        data_train = np.copy(X_norm)
        data_train = np.delete(data_train, index, 1)
        row_cor = row
        row_cor = np.delete(row_cor, index) # at the analyzed row, we delete the nan values
        mean_cor = np.delete(mean, index)
        st_dv_cor = np.delete(st_dv, index)
        row_cor = (row_cor - mean_cor) / st_dv_cor

        for k in range(c):
            p = index[k]
            y_train = X_norm[:, p].copy()
            w = Ridge(data_train, y_train,  10)
            yhat = np.dot(row_cor, w)

            row[p] = yhat * st_dv[p] + mean[p]
            if p in range(4, 9) or p in range(18, 25):  # cat features
                row[p] = np.round(row[p])

#MAKING THE DECISION TREE
del featNames[24]

data = np.delete(X_to_cor, 24, 1)
target = X_to_cor[:, 24]
clf = tree.DecisionTreeClassifier("entropy")
clf = clf.fit(data, target)
classNames = ['notckd', 'ckd']
dot_data = tree.export_graphviz(clf, out_file="Tree.dot", feature_names=featNames, class_names=classNames,
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('dtree_render', view=True)
