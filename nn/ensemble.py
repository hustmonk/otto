#!/usr/bin/env python
# -*- coding: GB2312 -*-
# Last modified: 

"""docstring
"""

__revision__ = '0.1'
files = ["my_neural_net_submission.csv0","my_neural_net_submission.csv1","my_neural_net_submission.csv2","my_neural_net_submission.csv3","my_neural_net_submission.csv4","my_neural_net_submission.csv5","my_neural_net_submission.csv6","my_neural_net_submission.csv7","my_neural_net_submission.csv8","my_neural_net_submission.csv9"]
fins = []
fout = open("sub.csv","w")
#for file in files:
for i in range(20):
    fin = open("result/my_neural_net_submission.csv" + str(i))
    head = fin.next()
    fins.append(fin)
fout.write(head)
for line in fins[0]:
    arr = line.strip().split(",")
    id = arr[0]
    preds = [[float(i) for i in arr[1:]]]
    for fin in fins[1:]:
        arr = fin.next().strip().split(",")
        preds.append([float(i) for i in arr[1:]])

    ps = []
    for i in range(len(arr[1:])):
        p = 0
        pd = []
        for j in range(len(fins)):
            p += preds[j][i]
            pd.append("%.2f" % preds[j][i])
        print pd
        ps.append("%.3f" % (p/len(fins)))
    fout.write("%s,%s\n" % (id,",".join(ps)))

fout.close()
