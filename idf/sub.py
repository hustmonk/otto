#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""
import logging
import logging.config
import sys
from datetime import datetime
import math
import sys
from math import exp, log, sqrt

__revision__ = '0.1'

def predict(x):
    return 1. / (1. + exp(-max(min(x, 35.), -35.)))

fout = open("sub.csv", "w")
fout.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n")
for line in open("train.txt.raw"):
    arr = line.strip().split(" ")
    id = arr[-1]
    arr = arr[:-1]
    pred = [predict(float(k.split(":")[1])) for k in arr]
    #pred = [k * k for k in pred]
    sum_pred = sum(pred)
    pred = ["%.2f" % (k/sum_pred) for k in pred]
    fout.write("%s,%s\n" % (id, ",".join(pred)))
fout.close()
