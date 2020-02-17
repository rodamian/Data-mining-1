from __future__ import division, print_function
import numpy as np
import argparse
import os
import numpy as np

dat = np.loadtxt('ECG200_TRAIN.txt', delimiter =',')

normal = dat[dat[:, 0] == 1, 1:len(dat)]
abnormal = dat[dat[:, 0] == -1, 1:len(dat)]
num_comparisons = len(normal) * len(abnormal)


### defining constrained DTW

t1=[1,2,3,4,2]
t2=[1,2,1,2,2]

def constrained_dtw(t1, t2, w):

    dist = np.zeros(shape=(len(t1), len(t2)))
    w = max(w, abs(len(t1)-len(t2)))

    acc_cost = np.zeros(shape=(len(t1), len(t2)))

    for i in range(len(t1)):
        for j in range(len(t2)):
            dist[i,j] = abs(t1[i]-t2[j])

        for j in range(max(0, i-w), min(len(t2), (i + w))):
            acc_cost[i,j] = dist[i,j] + min(acc_cost[i-1,j], acc_cost[i,j-1], acc_cost[i-1,j-1])
    path = [[len(t1)-1, len(t2)-1]]
    cost = 0
    i = len(t2)-1
    j = len(t1)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if acc_cost[i-1, j] == min(acc_cost[i-1, j-1], acc_cost[i-1, j], acc_cost[i, j-1]):
                i = i - 1
            elif acc_cost[i, j-1] == min(acc_cost[i-1, j-1], acc_cost[i-1, j], acc_cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])

    for [t1, t2] in path:
        cost = cost + acc_cost[t1,t2]
    return cost


for i in range(len(normal)):
    for j in range(len(abnormal)):

        avg_dist = np.zeros(4, dtype=float)

        avg_dist[0] = constrained_dtw(normal[i], abnormal[j], 0)
        avg_dist[1] = constrained_dtw(normal[i], abnormal[j], 10)
        avg_dist[2] = constrained_dtw(normal[i], abnormal[j], 25)
        avg_dist[3] = constrained_dtw(normal[i], abnormal[j], 30000)

print(avg_dist)

def manhattan_dist(t1,t2):
    return sum(abs(a-b) for a,b in zip(t1,t2))
