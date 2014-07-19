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
            biLabel[l] = (data == l)
        for l in biclassifier:
            print "training,using label %s."%str(l)
            biclassifier[l].fit(data,biLabel)
            print done
    
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        print "start classifying %s."%k
        resTmp = {}
        for l in biclassifier:
            print "classifying,using label %s."%str(l)
            resTmp[l] = np.array(biclassifier[l].predict(testings[k]),dtype = np.int)
            print "done."
        result = np.zeros(resTmp[0].shape)
        for l in resTmp:
            result[resTmp[l] == l] = l
        np.savetxt("./result/%s.seg"%k,result,fmt = "%d")
        
        
            