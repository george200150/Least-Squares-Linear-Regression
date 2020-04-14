'''
Created on 7 apr. 2020

@author: George
'''
import csv
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error



def loadData(fileName, inputVariabName, outputVariabName):
    size = len(inputVariabName)
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariables = [dataNames.index(inputVariabName[i]) for i in range(size)]
    
    inputs = [[float(data[i][selectedVariables[j]]) for j in range(size)] for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]
    
    return inputs, outputs


def plotData3D(inputs, outputs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i,o in zip(inputs,outputs):
        xs = i[0]
        ys = i[1]
        zs = o
        ax.scatter(xs, ys, zs, marker='o')
    
    ax.set_xlabel('GDP capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('happiness')
    
    plt.title('GDP capita vs. Freedom vs. happiness')
    plt.show()





def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()




def plotModel(feature1train, feature2train, trainOutputs, xref1, xref2, yref):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i1,i2,o in zip(feature1train,feature2train,trainOutputs):
        xs = i1
        ys = i2
        zs = o
        ax.scatter(xs, ys, zs, marker='o')
    
    ax.set_xlabel('GDP capita')
    ax.set_ylabel('Freedom')
    ax.set_zlabel('happiness')
    
    
    ax.plot(xref1, xref2, yref, label='parametric curve')
    plt.title('GDP capita vs. Freedom vs. happiness')
    plt.show()

    ####################################################################










import regression
from random import seed

# *two hundreds plots later*
def fun():
    seed(1)
    crtDir =  os.getcwd()
    filePath = os.path.join(crtDir, 'date.txt')
    
    inputs, outputs = loadData(filePath, ['Economy..GDP.per.Capita.', 'Freedom'], 'Happiness.Score')
    print('in:  ', inputs[:5])
    print('out: ', outputs[:5])
    
    firsts = [i for i,_ in inputs]
    seconds = [j for _,j in inputs]
    
    #plotDataHistogram(firsts, 'Capita GDP')
    #plotDataHistogram(seconds, 'Freedom')
    #plotDataHistogram(outputs, 'Happiness score')
    
    #plotData3D(inputs, outputs)
    
    
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
    
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    
    testSample = [i for i in indexes  if not i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample] 
    
    
    xx = [[el,el2] for el,el2 in trainInputs]
    
    
    regressor = linear_model.LinearRegression()
    #regressor = regression.MyLinearUnivariateRegression()
    regressor.fit(xx, trainOutputs) # FIT SINGLE MATRIX of noSamples x noFeatures
    
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[0] 
    
    feature1 = [el for el,_ in trainInputs]
    feature2 = [el2 for _,el2 in trainInputs]
    
    feature1train = [inputs[i][0] for i in trainSample]
    feature2train = [inputs[i][1] for i in trainSample]
    
    
    noOfPoints = 50
    xref1 = []
    val = min(feature1)
    step1 = (max(feature1) - min(feature1)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1
    
    
    xref2 = []
    val = min(feature2)
    step2 = (max(feature2) - min(feature2)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref2.append(val)
        val += step2
            
    yref = [w0 + w1 * el1 + w2 * el2 for el1, el2 in zip(xref1, xref2)]
    
    plotModel(feature1train, feature2train, trainOutputs, xref1, xref2, yref)
    
    
    
    
    
    computedTestOutputs = regressor.predict([[x,y] for x,y in testInputs])
    
    feature1 = [el for el,_ in testInputs]
    feature2 = [el2 for _,el2 in testInputs]
    
    feature1test = [inputs[i][0] for i in testSample]
    feature2test = [inputs[i][1] for i in testSample]
    
    
    noOfPoints = 50
    xref1 = []
    val = min(feature1)
    step1 = (max(feature1) - min(feature1)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref1.append(val)
        val += step1
    
    
    xref2 = []
    val = min(feature2)
    step2 = (max(feature2) - min(feature2)) / noOfPoints
    for _ in range(1, noOfPoints):
        for _ in range(1, noOfPoints):
            xref2.append(val)
        val += step2
    
    
    plotModel(feature1test, feature2test, computedTestOutputs, xref1, xref2, yref) # "predictions vs real test data"
    
    
    #compute the differences between the predictions and real outputs
    error = 0.0
    for t1, t2 in zip(computedTestOutputs, testOutputs):
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print("prediction error (manual): ", error)
    
    error = mean_squared_error(testOutputs, computedTestOutputs)
    print("prediction error (tool): ", error)


if __name__ == '__main__':
    fun()
