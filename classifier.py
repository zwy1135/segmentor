# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:50:16 2015

@author: wy
"""

from __future__ import print_function
import pickle

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

dumped = "clf.pickle"

class Classifier:
    def __init__(self, scaler=None, selector=None, clf=None):
        self.scaler = scaler or StandardScaler()
        self.selector = selector or VarianceThreshold(threshold = 0.85*(1-0.85))
        self.clf = clf or ExtraTreesClassifier(n_estimators=100,
                                               verbose=1,
                                               n_jobs=10, 
                                               class_weight="auto",
                                               max_depth=10)
                                               
    def preprocess(self, data, toFit=False):
        #impulse
        data[data == np.inf] = 1
        data[data == -np.inf] = -1
        data[np.isnan(data)] = 0
        
        if toFit:
            self.scaler.fit(data)
            self.selector.fit(data)
        
        data = self.scaler.transform(data)
        data = self.selector.transform(data)
        
        return data
        
    def fit(self, data, label):
        self.clf.fit(data, label)
        
    def predict(self, data):
        return self.clf.predict(data)
        
    def predict_proba(self, data):
        return self.clf.predict_proba(data)
        
    def transform(self,data,**kwargs):
        return self.clf.transform(data, **kwargs)
        
    def save(self, name = dumped):
        with open(name,"wb") as f:
            pickle.dump(self, f)
            
    def load(self, name = dumped):
        with open(name,"rb") as f:
            return pickle.load(f)
            
if __name__=="__main__":
    from model import Model
    clf = Classifier()
    name = "F0001A1EnCtrimR0"
    m = Model(name,True)
    data, label = m.feature, m.labels
    data = clf.preprocess(data,True)
    clf.fit(data, label)
    res1 = clf.predict(data)
    clf.save()
    clf = Classifier()
    clf = clf.load()
    res2 = clf.predict(data)
    print(res1==res2)
            
            
        