# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 21:13:17 2014

@author: wy
"""

from sklearn import svm
from dataBuilder import *
import numpy as np
import os
import cPickle

featureIndex = [1,2,4,7,11]

if __name__ == '__main__':
    if 'classifier.svm' in os.listdir('./'):
        print 'No needed for training'
        f = file('classifier.svm')
        classifier = cPickle.load(f)
        f.close()
    else:
        classifier = svm.SVC()
        data,label = dataAndLabel()
        data = data.T[featureIndex].T
        data[data == np.inf] = 1
        data[data == -np.inf] = -1
        data[np.isnan(data)] = 0
        print 'Training classifier.'
        classifier.fit(data,label)
        print 'Trained classifier.'
        f = file('classifier.svm','w')
        cPickle.dump(classifier,f)
        f.close()
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
        np.savetxt('./result/%s.seg'%k,result,fmt='%d')
