# python3
# coding: utf-8

import sys
import numpy as np
from scipy.stats import ttest_ind as test

file0 = sys.argv[1]
file1 = sys.argv[2]

prec0 = []
prec1 = []

recall0 = []
recall1 = []

f0 = []
f1 = []

for line in open(file0, 'r').readlines()[1:-1]:
    res = line.strip().split('\t')
    precision = float(res[1])
    recall = float(res[2])
    fscore = float(res[3])

    prec0.append(precision)
    recall0.append(recall)
    f0.append(fscore)

for line in open(file1, 'r').readlines()[1:-1]:
    res = line.strip().split('\t')
    precision = float(res[1])
    recall = float(res[2])
    fscore = float(res[3])

    prec1.append(precision)
    recall1.append(recall)
    f1.append(fscore)

print('Comparing', file0, file1)
print('Precisions')
print(np.average(prec0), np.average(prec1))
print('T-test:', test(prec0, prec1))


print('Recalls')
print(np.average(recall0), np.average(recall1))
print('T-test:', test(recall0, recall1))


print('F-scores')
print(np.average(f0), np.average(f1))
print('T-test:', test(f0, f1))





