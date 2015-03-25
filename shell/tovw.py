#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""

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
        for i in arr[1:]:
            if float(i) > 10:
                ks.append("X")
            else:
                ks.append(i)
        feature = " ".join( [ str(i)+"_"+ks[i] for i in range(len(ks))])
        featurev = " ".join( [ str(i)+":"+arr[i] for i in range(1, len(arr))])
        fout.write("%s %s|f %s |v %s\n" % (label, id, feature, featurev))
    fout.close()

write("../data/train.csv.shuffle", "train.txt", True)
write("../data/test.csv", "test.txt", False)
write("../data/train.csv1.shuffle", "train.txt1", True)
write("../data/train.csv2", "train.txt2", True)
