#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""
import math

__revision__ = '0.1'

def write(fin, fout, train):
    fin = open(fin)
    fout = open(fout, "w")
    head = fin.next().strip().split(",")
    for line in fin:
        arr = line.strip().split(",")
        label = "1"
        if train:
            label = arr[-1][-1]
            arr = arr[:-1]
        id = arr[0]
        ks = []
        for i in range(1, len(arr)):
            if float(arr[i]) > 0:
                ks.append([i, math.sqrt(1+math.log(float(arr[i]) + 1.1))])
                
        #feature = " ".join( [ str(k[0])+"_1" for k in ks])
        featurev = [ "%d:%.3f" % (k[0], k[1]) for k in ks]
        """
        for k1 in ks:
            for k2 in ks:
                if k1[0] > k2[0]:
                    featurev.append("%d_%d:%.3f" % (k1[0],k2[0], k1[1]*k2[1]))
        """
        featurev = " ".join(featurev)
        #fout.write("%s %s|f %s |v %s\n" % (label, id, feature, featurev))
        fout.write("%s %s|v %s\n" % (label, id, featurev))
    fout.close()

#write("../data/train.csv.shuffle", "train.txt", True)
#write("../data/test.csv", "test.txt", False)
write("../data/train.csv1.shuffle", "train.txt1", True)
write("../data/train.csv2", "train.txt2", True)
