#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------
@brief	expTable介绍
@author	jungle
@date	2016/01/05
---------------------------------------
'''

import math
import matplotlib.pyplot as plt

EXP_TABLE_SIZE = 1000
MAX_EXP = 6

expTable = [0 for i in range(0, EXP_TABLE_SIZE)]

print len(expTable)

# 初始化。
for i in range(0, EXP_TABLE_SIZE):
    expTable[i] = math.exp((i * 1.0 / EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
    expTable[i] = expTable[i] / (expTable[i] + 1)

print expTable

## 此处f 范围(-6,6)用f/1000.0来代替. 
m = [int((f/1000.0 + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)) for f in range(-6000, 6000)]

print m


result = [expTable[x] for x in m]
#print result

fig = plt.figure()

ax = plt.subplot(1,1,1)

## 设置坐标范围
#plt.ylim(ymax=1.2)
#plt.ylim(ymin=0.5)

ax.plot(result)

#ax.plot(expTable)
#ax.plot(expTable, label="exp")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])

# python自动取整.

plt.show()



