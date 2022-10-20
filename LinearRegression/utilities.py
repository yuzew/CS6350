# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:10:28 2022

@author: Yuze
"""

def createInputMatrices(data, labelCol):
    x = []
    y = []
    for example in data:
        exampleList = [1] # one in front for the bias term
        for attr in example:
            if attr == labelCol:
                y.append(float(example[attr]))
            else:
                exampleList.append(float(example[attr]))
        x.append(exampleList)
    return x, y

#####
# Purpose: Find the dot product between w and x
#####
def dot(w, x):
    return sum([wk*xk for wk,xk in zip(w,x)])

#####
# Purpose: Calculated the cost function value for the given vector w, x, and y
#####
def costValue(w,x,y):
    costSum = 0
    for i in range(len(y)):
        costSum += (y[i] - dot(w,x[i]))**2
    return 0.5*costSum