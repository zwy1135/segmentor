# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 22:23:01 2014

@author: zeng
"""

from sklearn.feature_selection import SelectKBest,f_classif
from dataBuilder import *
import numpy as np

if __name__ == "__main__":
    data,label = dataAndLabel()
    data[data == np.inf]=1
    data[data == -np.inf]=-1
    data[np.isnan(data)] = 0
    print data
    selector = SelectKBest(f_classif,5)
    selector.fit(data,label)
    res = selector.get_support(True)
    print res
    #np.savetxt("featureIndex.txt",fmt="%d")
    
    