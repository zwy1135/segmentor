# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 22:23:01 2014

@author: zeng
"""
from __future__ import print_function

from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.ensemble import ExtraTreesClassifier
from dataBuilder import *
import numpy as np

def kBestSelect():
    data,label = dataAndLabel()
    data[data == np.inf]=1
    data[data == -np.inf]=-1
    data[np.isnan(data)] = 0
    #print data
    selector = SelectKBest(f_classif,5)
    selector.fit(data,label)
    featureIndex = selector.get_support(True)
    
    return featureIndex
    #np.savetxt("featureIndex.txt",fmt="%d")
    
def treeSelect():
    data,label = dataAndLabel()
    data[data == np.inf]=1
    data[data == -np.inf]=-1
    data[np.isnan(data)] = 0
    clf = ExtraTreesClassifier(n_estimators = 15,max_features = None)
    clf.fit(data)
    featureIndex = list(np.argsort(clf.feature_importances_))
    featureIndex.reverse()
    featureIndex = featureIndex[:5]
    
    return featureIndex
    
if __name__ == "__main__":
    print_function(treeSelect())
    
    
    