# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 22:48:49 2014

@author: wy
"""

from pylab import *
import os
import numpy as np

def main(segPath = "./seg",resultPath = "./result",trainSet = [1,2,3,4,5,6,8,11,14,20]):
    sfilelist = os.listdir(segPath)
    rfilelist = os.listdir(resultPath)
    trainSet = np.array(trainSet)
    dataset = {}

    for filename in sfilelist:
        if filename in rfilelist:
            dataset[filename] = [np.loadtxt(os.path.join(segPath,filename)),np.loadtxt(os.path.join(resultPath,filename))]

    figure()
    trainSet = map(lambda x:str(x)+'.seg',trainSet)
    result = []
    for name in trainSet:
        result.append(sum(dataset[name][0]==dataset[name][1])/float(len(dataset[name][0])))

    for i in range(len(result)):
        bar(i,result[i])
        text(i,result[i],'%.3f'%result[i])
    xticks(np.arange(len(trainSet))+0.5,trainSet)
    title(u"Accuracy rate of training-set segmentation")
    ylabel(u"Accuracy rate")
    
    figure()
    result = []
    testSet = [name for name in dataset if name not in trainSet]
    testSet.remove('16.seg')
    testSet.remove('18.seg')
    testSet.sort()
    for name in testSet:
        result.append(sum(dataset[name][0]==dataset[name][1])/float(len(dataset[name][0])))
    
    for i in range(len(result)):
        bar(i,result[i])
        text(i,result[i],'%.3f'%result[i])
    xticks(np.arange(len(testSet))+0.5,testSet)
    title(u"Accuracy rate of testing-set segmentation")
    ylabel(u"Accuracy rate")
    show()
        
       
    


if __name__ == '__main__':
    main()

