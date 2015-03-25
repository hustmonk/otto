#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""

__revision__ = '0.1'

fin = open("train.csv")

header = fin.next().strip().split(",")
counts = []
for i in range(len(header)):
    counts.append({})
for line in fin:
    arr = line.strip().split(",")
    for i in range(1, len(header)):
        key = arr[i]
        counts[i][key] = counts[i].get(key, 0) + 1

#for i in range(1, len(header)):
#    print header[i],len(counts[i]),counts[i]
for i in range(1, len(header)):
    for (k,v) in counts[i].items():
        print k,v
