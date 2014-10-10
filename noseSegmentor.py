# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 21:46:18 2014

@author: wy
"""

from __future__ import print_function
import os
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dataBuilder import *


def noseSegment():
    if "pipe.nose" in os.listdir():
        print("No need for training.")
        with open("pipe.nose","rb") as f:
            pipe = pickle.load(f)
            
    else:
        classifier = ExtraTreesClassifier(n_estimators=15,verbose=1)
        scaler = StandardScaler()
        pipe = Pipeline([("scaler",scaler),("classifier",classifier)])
        data,label = dataAndLabel()
        data[data == np.inf] = 1
        data[data == -np.inf] = -1
        data[np.isnan(data)] = 0
        print("training.")
        pipe.fit(data,label)
        print("done. saving data.")
        with open("pipe.nose","wb") as f:
            pickle.dump(pipe,f)
            
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        print( 'classifying %s'%k)
        data_k = testings[k]#.T[featureIndex].T
        data_k[data_k == np.inf] = 1
        data_k[data_k == -np.inf] = -1
        data_k[np.isnan(data_k)] = 0
        result = pipe.predict(data_k)
        print( 'classified.')
        result = np.array(result,dtype = np.int)
        print( 'saving result.')
        np.savetxt('./result/%s.seg'%k,result,fmt='%d')
        
if __name__ == "__main__":
    noseSegment()
        
        
    


