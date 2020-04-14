'''
Created on 7 apr. 2020

@author: George
'''

import numpy as np
# from sklearn import linear_model
# from sklearn.linear_model.stochastic_gradient import SGDRegressor
# from sklearn.metrics import mean_squared_error, r2_score

from math import exp
from math import log2
from numpy.linalg import inv
# from random import shuffle, random
# from logisticRegression import SGDLogisticTool, myLogisticRegression
 
class MyLinearUnivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0
    # learn a linear univariate regression model by using training inputs (x) and outputs (y) 
    def fit2(self, x, y):
        sx = sum(x)
        sy = sum(y)
        sx2 = sum(i * i for i in x)
        sxy = sum(i * j for (i,j) in zip(x, y))
        w1 = (len(x) * sxy - sx * sy) / (len(x) * sx2 - sx * sx)
        w0 = (sy - w1 * sx) / len(x)
        self.intercept_, self.coef_ =  w0, w1
    # predict the outputs for some new inputs (by using the learnt model)
    
    def fit(self, x, y):
        sizeX = len(x) # sizeX is the p+1 length of the samples + the vector of ones 
        X = []
        
        for i in range(0,sizeX): # create the 1n which is an n x 1 vector of ones
            X.append([1,x[i][0],x[i][1]])
        
        # y    =          X          *   b  +  e
        #
        # (y1)   (1 x11 x12 ... x1p)   (b1)   (e1)
        # (y2)   (1 x21 x22 ... x2p)   (b2)   (e2)
        # (y3)   (1 x31 x32 ... x3p)   (b3)   (e3)
        # (..) = (. ... ... \.. ...) * (..) + (..)
        # (..)   (. ... ... .\. ...)   (..)   (..)
        # (..)   (. ... ... ..\ ...)   (..)   (..)
        # (yn)   (1 ... xn2 ... xnp)   (bp)   (en)
        
        # y = beta0 + beta1 * x1 + beta2 * x2
        
        
        #http://mathforcollege.com/nm/mws/gen/06reg/mws_gen_reg_spe_multivariate.pdf
        #
        #The single variable regression, setting up sum of squares of the residuals,
        #SRm = sum i=1..6 (yi - beta0 - beta1 * x1i - beta2 * x2i) ^ 2                 (4)
        #and differentiating with respect to each unknown coefficient arid equating each
        #partial derivative to zero,
        #drond SRm / drond beta0 = -2 sum (yi - beta0 - beta1 * x1i - beta2 * x2i) = 0 (5)
        #drond SRm / drond beta1 = -2 sum x1i * (yi - beta0 - beta1 * x1i - beta2 * x2i) = 0 (6)
        #drond SRm / drond beta2 = -2 sum x2i * (yi - beta0 - beta1 * x1i - beta2 * x2i) = 0 (7)
        #
        #we obtain the following matrix expression:
        #(    n         sum(x1i)     sum(x2i)        )   (beta0)   (   sum(yi)   )
        #( sum(x1i)    sum(x1i^2)    sum(x2i * x1i)  ) * (beta1) = (sum(x1i * yi))       (8)
        #( sum(x2i)  sum(x1i * x2i)  sum(x2i^2)      )   (beta2)   (sum(x2i * yi))
        
        n = len(x)
        a1 = [n, sum([x[i][0] for i in range(n)]), sum([x[i][1] for i in range(n)])]
        a2 = [sum([x[i][0] for i in range(n)]), sum([x[i][0]**2 for i in range(n)]), sum([x[i][0]*x[i][1] for i in range(n)])]
        a3 = [sum([x[i][1] for i in range(n)]), sum([x[i][0]*x[i][1] for i in range(n)]), sum([x[i][1]**2 for i in range(n)])]
        
        A = np.array([a1,a2,a3])
        
        b1 = sum([y[i] for i in range(n)])
        b2 = sum([x[i][0] * y[i] for i in range(n)])
        b3 = sum([x[i][1] * y[i] for i in range(n)])
        
        b = np.array([b1, b2, b3])
        
        betas = np.linalg.solve(A, b)
        betas = list(betas)
        
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
        
        #SRm = sum((y[i] - b[0] - b[1] * x[0][i] - b[2] * x[1][i])**2 for i in range(n))
        #derivSrmWRespectToBeta0 = (-2) * sum(y[i] - b[0] - b[1] * x[0][i] - b[2] * x[1][i]) # this equations equals to 0
        #derivSrmWRespectToBeta1 = (-2) * sum(x[0][i] * (y[i] - b[0] - b[1] * x[0][i] - b[2] * x[1][i])) # this equations equals to 0
        #derivSrmWRespectToBeta2 = (-2) * sum(x[1][i] * (y[i] - b[0] - b[1] * x[0][i] - b[2] * x[1][i])) # this equations equals to 0
        
        '''b = [0,0,0] # w0, w1, w2
        n = len(x)
        p = len(x[0])
        
        smmm = 0
        for i in range(n):
            smm = 0
            for j in range(p):
                smm += b[j] * x[i][j]
            smmm += y[i] - b[0] - smm**2
        
        Frobenius_norm = min ( sum(y[i] - b[0] - (sum(b[j] * x[i][j] for j in range(p)))**2 for i in range(n)) )'''
    
    
    # predict the outputs for some new inputs (by using the learnt model)
    def predict2(self, x):
        if (isinstance(x[0], list)):
            return [self.intercept_ + self.coef_ * val[0] for val in x]
        else:
            return [self.intercept_ + self.coef_ * val for val in x]
    
    
    def predict(self, x):
        if (isinstance(x[0], list)):
            return [self.intercept_ + self.coef_[0] * val[0] + self.coef_[1] * val[1] for val in x]
        else:
            return [self.intercept_ + self.coef_ * val for val in x]