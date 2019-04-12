#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import random

## 词频
#wordfreq = [0.01, 0.002, 0.38, 0.32, 0.02, 0.03, 0.001, 0.05, 0.01, 0.01, 0.02, 0.03, 0.008, 0.10]

wordfreq = [ 0.001 * i for i in range(1, 100)]
wordfreq2 = [i for i in range(1, 100)]

##
sampling = 1e-3
next_random = 1


# subsampling method 1.
def subsampling_prob(freq, sampling):
    return 1 - math.sqrt(sampling / freq)


# subsampling method 2.
def subsampling_prob2(freq, sampling):
    return 1 - (math.sqrt(sampling / freq) + sampling / freq)


# random subsampling method 3:
def subsampling_prob3(freq, sampling):
    global next_random
    ran = (math.sqrt(sampling / freq) + sampling / freq)

    next_random = (next_random * 25214903917 + 11) & 0xFFFF

    if ran < next_random/65535:
        return 0
    else:
        return 1


def subsampling_prob4(freq, sampling):
    ran = (math.sqrt(sampling / freq) + sampling / freq)
    return ran


fig = plt.figure()


def test1():
    a = [subsampling_prob4(i, 1e-3) for i in wordfreq]
    b = [subsampling_prob4(i, 1e-4) for i in wordfreq]
    c = [subsampling_prob4(i, 1e-5) for i in wordfreq]
    d = [subsampling_prob4(i, 1e-6) for i in wordfreq]
    zero = [0 for i in wordfreq]


    ax = plt.subplot(1, 1, 1)

    ## 设置坐标范围
    #plt.ylim(ymax=3)
    #plt.ylim(ymin=-3)

    ## 绘制4条曲线.
    #ax.plot(wordfreq2, label="wordfreq")
    ax.plot(a, label="1:subsample 1e-3")
    ax.plot(b, label="2:subsample 1e-4")
    ax.plot(c, label="3:subsample 1e-5")
    ax.plot(d, label="4:subsample 1e-6")
    ax.plot(zero, label="zero")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    plt.show()


# ---------------------------------
# test2: 随机生成归一化概率.
# ---------------------------------
def test2():
    r = [random.random() for i in range(1, 10)]
    s = sum(r)
    wordfreq = [i / s for i in r]
    a = [subsampling_prob(i, sampling) for i in wordfreq]
    b = [subsampling_prob2(i, sampling) for i in wordfreq]

    print(wordfreq)

    plt.plot(wordfreq)
    plt.plot(a)
    plt.plot(b)

    plt.show()

if __name__ == '__main__':
    test1()