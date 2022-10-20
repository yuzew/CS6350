# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:08:41 2022

@author: Yuze
"""

import utilities

#####
# Purpose: Perform the stochastic gradient descent learning algorithm
#####
def StochasticGradientDescent(x, y, r, iterations):
    import random

    wghts = [0 for i in range(len(x[0]))]
    costs = [utilities.costValue(wghts, x, y)]
    converge = False

    for i in range(iterations):
        # randomly sample an example
        index = random.randrange(len(x))
        # update weight vector with the stochastic grad
        newWghts = []
        for j in range(len(wghts)):
            newWghts.append(wghts[j] + r*x[index][j]*(y[index] - utilities.dot(wghts,x[index])))
        wghts = newWghts
        # check convergence (calculate cost function)
        costVal = utilities.costValue(wghts, x, y)
        if abs(costVal - costs[-1]) < 10e-6:
            converge = True
        costs.append(costVal)

    return wghts, costs, converge