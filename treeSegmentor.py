# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 16:30:35 2014

@author: wy
"""

import os,cPickle
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from dataBuilder import *

featureIndex = [1,2,4,7,11]

if __name__=="__main__":
    if "tree" in os.listdir("./"):
        print "No need to train"
        with open("tree") as f:
            classifier = cPickle.load(f)
    else:
        classifier = ExtraTreesClassifier(n_estimators=15,max_features = None)
        data,label = dataAndLabel()
        data = data.T[featureIndex].T
        data[data == np.inf] = 1
        data[data == -np.inf] = -1
        data[np.isnan(data)] = 0
        print "training classifier"
        classifier.fit(data,label)
        print "Saving result."
        with open("tree","w") as f:
            cPickle.dump(classifier,f)
            
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        print 'classifying %s'%k
        data_k = testings[k].T[featureIndex].T
        data_k[data_k == np.inf] = 1
        data_k[data_k == -np.inf] = -1
        data_k[np.isnan(data_k)] = 0
        result = classifier.predict(data_k)
        print 'classified.'
        result = np.array(result,dtype = np.int)
        print 'saving result.'
        np.savetxt('./result_tree/%s.seg'%k,result,fmt='%d')
            