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

def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def predict(x):
    p = 1. / (1. + exp(-max(min(x, 35.), -35.)))
    if p < 0.01:
        p = 0.01
    return p

def getpredict(arr):
    pred = [predict(float(k.split(":")[1])) for k in arr]
    pred = [k * k for k in pred]
    sum_pred = sum(pred)
    pred = [k/sum_pred for k in pred]
    return pred

if __name__ == "__main__":
    for i in range(100):
        loss = 0
        for j in range(10):
            loss += logloss(i/100.0, 0)
        loss += logloss(i/100.0, 1)
        print i,loss
