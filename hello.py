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

result = [math.exp(-i/10) for i in range(0, 28)]


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




## 如果是从i的角度去理解.

plt.show()



