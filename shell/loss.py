#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""
from common import *
import logging
import logging.config
import sys
from datetime import datetime
import math
import sys
from math import exp, log, sqrt

logging.config.fileConfig("log.conf")

__revision__ = '0.1'
handler = logging.handlers.RotatingFileHandler("logs", maxBytes = 1024*1024, backupCount = 5)
logger = logging.getLogger("example")
logger.addHandler(handler)

ys = []
for line in open("train.txt2"):
    ys.append(int(line[0]))

i = 0
loss = 0
for line in open("train.txt2.raw"):
    arr = line.strip().split(" ")[:-1]
    pred = getpredict(arr)
    #k = sum(pred[1:3])/2
    #pred[1],pred[2] = k, k
    k = 0
    for j in range(len(pred)):
        if j == ys[i] - 1:
            loss += logloss(pred[j], 1)
            k = logloss(pred[j], 1)
            if k > 1:
                print k, pred, j
        #else:
        #    loss += logloss(pred[j], 0)
    i = i+1

logger.info(loss/len(ys))
