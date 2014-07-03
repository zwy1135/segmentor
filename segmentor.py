# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 21:13:17 2014

@author: wy
"""

from sklearn import svm
from dataBuilder import *
import numpy as np

if __name__ == '__main__':
    classifier = svm.SVC()
    data,label = dataAndLabel()
    print 'Training classifier.'
    classifier.fit(data,label)
    print 'Trained classifier.'
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        result = classifier.predict(testing[k])
        np.savetxt('./result/%s.seg',result)