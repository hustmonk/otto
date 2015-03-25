#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""

__revision__ = '0.1'
import random

def shuffle(fin, fout, bl):
    fin = open(fin)
    fout = open(fout, "w")
    fout.write(fin.next())
    items = []
    for i in fin:
        items.append(i)

    random.shuffle(items)
    for i in items:
        fout.write(i)
    random.shuffle(items)
    for i in items:
        fout.write(i)
    random.shuffle(items)
    for i in items:
        fout.write(i)


shuffle("../data/train.csv", "train.csv.shuffle",True)
#shuffle("../data/test.csv", "test.csv.shuffle",True)
shuffle("../data/train.csv1", "train.csv1.shuffle", True)
#shuffle("../data/train.csv2", "train.csv2.shuffle", True)
        
