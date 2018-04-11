# -*- coding: utf-8 -*-
"""
Spyder Editor 

@author vanish

This is a temporary script file.
"""

from numpy import *
from os import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import array

def createDataSet():
    group = array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat= diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.9
    datingDataMat,datingLabels = file2matrix('G:\machine_learning\Ch02\datingTestSet2.txt')
    normMatSet,ranges,minVals = autoNorm(datingDataMat)
    m = normMatSet.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMatSet[i,:],normMatSet[numTestVecs,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with: %d, the real answer is :%d.'%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]): errorCount+=1
    print('the total error rate is:%f'%(errorCount/float(numTestVecs)))
    
def classifyperson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filter miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per years?"))
    datingDataMat,datingLabels = file2matrix('G:\machine_learning\Ch02\datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('you will probably like this person:',resultList[classifierResult-1])
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    trainingFileList = listdir('G:\machine_learning\Ch02\\trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('G:\machine_learning\Ch02\\trainingDigits\\%s'%(fileNameStr))
    testFileList = listdir('G:\machine_learning\Ch02\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('G:\machine_learning\Ch02\\testDigits\\%s'%fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classfier came back with:%d,the real answer is:%d'%(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):errorCount+=1.0
    print('\nthe total number of errors is: %d'%(errorCount))
    print("\nthe total error rate is: %f"%(errorCount/float(mTest)))