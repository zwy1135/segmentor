# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 21:46:18 2014

@author: wy
"""

from __future__ import print_function
import os
import pickle
import time


import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dataBuilder import *
from graphcut import buildDiGraph,cutAndLabel


def noseSegment():
    print("started at %s"%str(time.localtime()))
    if "pipe.nose" in os.listdir():
        print("No need for training.")
        with open("pipe.nose","rb") as f:
            pipe = pickle.load(f)
            
    else:
        classifier = ExtraTreesClassifier(n_estimators=100,verbose=1,n_jobs=10)
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
        print("training time end at %s"%str(time.localtime()))
        
    faceMap,vertexMap = faceMapping('./testing')       
    testings = getFeatureOfFace((faceMap,vertexMap))
    for k in testings:
        print( 'classifying %s'%k)
        data_k = testings[k]#.T[featureIndex].T
        data_k[data_k == np.inf] = 1
        data_k[data_k == -np.inf] = -1
        data_k[np.isnan(data_k)] = 0
#        result = pipe.predict(data_k)
#        print( 'classified.')
#        result = np.array(result,dtype = np.int)
#        print( 'saving result.')
#        np.savetxt('./result/%s.seg'%k,result,fmt='%d')
        print("compute probability.")
        proba = pipe.predict_proba(data_k)
        print("saving prob")
        np.savetxt("./result_prob/%s.prob"%k,proba)
        
        DG,size = buildDiGraph(faceMap[k],pipe.transform(data_k),proba)
        print("cutting.")
        result = cutAndLabel(DG,size)
        print( 'saving result.')
        np.savetxt('./result/%s.seg'%k,result,fmt='%d')        
    print("end at %s"%str(time.localtime()))
        
        
if __name__ == "__main__":
    noseSegment()
        
        
    


