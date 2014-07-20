# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:44:17 2014

@author: wy
"""

from sklearn import svm
from dataBuilder import *
import numpy as np
import os
import cPickle

if __name__ == '__main__':
    """
    Train svm for each label,classify it then merge result
    """
    if 'biclassifier.svm' in os.listdir('./'):
        print "No need for training."
        with open('biclassifier.svm') as f:
            biclassifier = cPickle.load(f)
            
    else:
        biclassifier = {}
        biLabel = {}
        data,label = dataAndLabel()
        labelSet = set(label)
        labelSet.remove(0)
        for l in labelSet:
            biclassifier[l] = svm.SVC()
            biLabel[l] = (label == l)
        for l in biclassifier:
            print "training,using label %s."%str(l)
            biclassifier[l].fit(data,biLabel[l])
            print "done"
        with open('biclassifier.svm','w') as f:
            cPickle.dump(biclassifier,f)
    
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        print "start classifying %s."%k
        resTmp = {}
        shape = 0
        for l in biclassifier:
            print "classifying,using label %s."%str(l)
            resTmp[l] = np.array(biclassifier[l].predict(testings[k]),dtype = np.bool)
            print "done."
            shape = resTmp[l].shape
        result = np.zeros(shape)
        for l in resTmp:
            result[resTmp[l]] = l
        np.savetxt("./result/%s.seg"%k,result,fmt = "%d")
        
        
            