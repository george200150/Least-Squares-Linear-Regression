'''
Created on 13 apr. 2020

@author: George
'''

'''
Created on 7 apr. 2020

@author: George
'''

#import numpy as np '''Fara NumPy'''
# from sklearn import linear_model
# from sklearn.linear_model.stochastic_gradient import SGDRegressor
# from sklearn.metrics import mean_squared_error, r2_score

#from math import exp
#from math import log2
#from numpy.linalg import inv
# from random import shuffle, random
# from logisticRegression import SGDLogisticTool, myLogisticRegression

        
def zeros_matrix(rows, cols): # create a zero matrix
    A = []
    for _ in range(rows):
        A.append([])
        for _ in range(cols):
            A[-1].append(0.0)

    return A

def copy_matrix(M): # deep copy a matrix
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC

def matrix_multiply(A,B): # mathematically multiply two matrices
    rowsA = len(A)
    colsA = len(A[0])

    rowsB = len(B)
    colsB = len(B[0])

    if colsA != rowsB:
        raise ValueError("Number of A columns must equal number of B rows.")

    C = zeros_matrix(rowsA, colsB)

    for i in range(rowsA):
        for j in range(colsB):
            total = 0
            for ii in range(colsA):
                total += A[i][ii] * B[ii][j]
            C[i][j] = total

    return C


def solve(A, B):
    '''
    @param A: coefficient matrix of the equations
    @param B: column vector having the results
    @return: list of solutions of the system of equations
    '''
    AA = copy_matrix(A) # python has memory issues
    n = len(A)
    BB = copy_matrix(B) # comparing to java...
     
    indices = list(range(n)) # allow flexible row referencing
    for FD in range(n): # FD stands for "FOCUS DIAGONAL" - interesting thing I found online
        FDScaler = 1.0 / AA[FD][FD]
        
        # FIRST: scale FD row with FD inverse. 
        for j in range(n): # Use j to indicate column looping.
            AA[FD][j] *= FDScaler
        BB[FD][0] *= FDScaler
         
        # SECOND: operate on all rows except FD row.
        for i in indices[0:FD] + indices[FD+1:]: # skip FD row.
            CRScaler = AA[i][FD] # cr stands for CURRENT ROW
            for j in range(n): # cr - CRScaler * FDRow.
                AA[i][j] = AA[i][j] - CRScaler * AA[FD][j]
            BB[i][0] = BB[i][0] - CRScaler * BB[FD][0]
    
    ret = [x[0] for x in BB] # convert output to desired type (this is one level too deep comparing to the linear_model)
    return ret

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
        
        A = [a1,a2,a3]
        
        b1 = sum([y[i] for i in range(n)])
        b2 = sum([x[i][0] * y[i] for i in range(n)])
        b3 = sum([x[i][1] * y[i] for i in range(n)])
        
        b = [[b1], [b2], [b3]] # simulate ndarray
        
        betas = solve(A, b) # aici facem treaba "de manuta"
        betas = list(betas)
        
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
        
        
    
    
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