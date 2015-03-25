#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""

__revision__ = '0.1'

"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle

__author__ : Abhishek Thakur
"""
import site
site.addsitedir('/Library/Python/2.7/site-packages')
import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing

# import data
train = pd.read_csv('../data/train.csv.shuffle')
test = pd.read_csv('../data/test.csv')
sample = pd.read_csv('../data/sampleSubmission.csv')

# drop ids and get labels
labels = train.target.values
train_ids = train.id.values
test_ids = test.id.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

def write(filename, x, y, ids):
    print type(x)
    fout = open(filename, "w")
    for i in range(len(x)):
        f = ["%d:%f" % (j, x[i][j]) for j in range(len(x[i])) if x[i][j] >0.0001]
        if len(y) == 0:
            fout.write("1 %s|f %s\n" %(ids[i], " ".join(f)))
        else:
            fout.write("%s %s|f %s\n" %(y[i][-1], ids[i], " ".join(f)))

write("train.txt", train, labels, train_ids )
write("test.txt", test, [], test_ids)
