# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 20:17:22 2015

@author: wy
"""

from __future__ import print_function
import os
import time
import pickle

import numpy as np

from model import Model
from model import dataPath, testPath

from classifier import Classifier, dumped

from graph import build_weight_graph,build_s_t_graph_list,cut_and_label



def main():
    print("started at %s"%str(time.localtime()))
    if dumped in os.listdir():
        with open(dumped,"rb") as f:
            clf = pickle.load(f)
    else:
        clf = Classifier()
        
        dataNameList = os.listdir(dataPath)
        
        trainModels = [Model(name.split(".")[0], True) for name in dataNameList]

        data = np.concatenate([m.feature for m in trainModels],axis=0)
        label = np.concatenate([m.labels for m in trainModels],axis=0)
        
        data = clf.preprocess(data, True)
        
        clf.fit(data, label)
        print("training time end at %s"%str(time.localtime()))
        clf.save()
        
    testNameList = os.listdir(testPath)
    testModels = [Model(name.split(".")[0]) for name in testNameList]
    for m in testModels:
        data = m.feature
        data = clf.preprocess(data)
        
        print("compute probability.")
        proba = clf.predict_proba(data)
        print("saving prob")
        np.savetxt("./result_prob/%s.prob"%m.name,proba)
        
        print("saving mid-res.")
        np.savetxt("./result_mid/%s.seg"%m.name,np.argmax(proba,axis=-1),fmt="%d")
        
        print("cutting")
        WG = build_weight_graph(m.faceGraph,clf.transform(data, threshold="median"))
        gList = build_s_t_graph_list(WG, proba)
        
        result = cut_and_label(gList)
        print( 'saving result.')
        np.savetxt('./result/%s.seg'%m.name,result,fmt='%d') 
        
    print("end at %s"%str(time.localtime()))
    
if __name__=="__main__":
    main()
        
        
    
    
    
    

