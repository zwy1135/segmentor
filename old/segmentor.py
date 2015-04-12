# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 21:13:17 2014

@author: wy
"""
from __future__ import print_function
import os
import pickle


from sklearn import svm
from dataBuilder import *
import numpy as np


featureIndex = [1,2,4,7,11]

def svmSegment():
    if 'classifier.svm' in os.listdir('./'):
        print( 'No needed for training')
        f = file('classifier.svm')
        classifier = pickle.load(f)
        f.close()
    else:
        classifier = svm.SVC()
        data,label = dataAndLabel()
        data = data.T[featureIndex].T
        data[data == np.inf] = 1
        data[data == -np.inf] = -1
        data[np.isnan(data)] = 0
        print( 'Training classifier.')
        classifier.fit(data,label)
        print( 'Trained classifier.')
        f = file('classifier.svm','w')
        pickle.dump(classifier,f)
        f.close()
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        print( 'classifying %s'%k)
        data_k = testings[k].T[featureIndex].T
        data_k[data_k == np.inf] = 1
        data_k[data_k == -np.inf] = -1
        data_k[np.isnan(data_k)] = 0
        result = classifier.predict(data_k)
        print( 'classified.')
        result = np.array(result,dtype = np.int)
        print( 'saving result.')
        np.savetxt('./result/%s.seg'%k,result,fmt='%d')
        
def treeSegment():
    if "tree" in os.listdir("./"):
        print( "No need to train")
        with open("tree") as f:
            classifier = pickle.load(f)
    else:
        classifier = ExtraTreesClassifier(n_estimators=15,max_features = None)
        data,label = dataAndLabel()
        data = data.T[featureIndex].T
        data[data == np.inf] = 1
        data[data == -np.inf] = -1
        data[np.isnan(data)] = 0
        print( "training classifier")
        classifier.fit(data,label)
        print( "Saving result.")
        with open("tree","w") as f:
            pickle.dump(classifier,f)
            
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        print( 'classifying %s'%k)
        data_k = testings[k].T[featureIndex].T
        data_k[data_k == np.inf] = 1
        data_k[data_k == -np.inf] = -1
        data_k[np.isnan(data_k)] = 0
        result = classifier.predict(data_k)
        print( 'classified.')
        result = np.array(result,dtype = np.int)
        print( 'saving result.')
        np.savetxt('./result_tree/%s.seg'%k,result,fmt='%d')
        
def svmBiSegmentor():
    """
    Train svm for each label,classify it then merge result
    """
    if 'biclassifier.svm' in os.listdir('./'):
        print( "No need for training.")
        with open('biclassifier.svm') as f:
            biclassifier = pickle.load(f)
            
    else:
        biclassifier = {}
        biLabel = {}
        data,label = dataAndLabel()
        data = data.T[featureIndex].T
        data[data == np.inf] = 1
        data[data == -np.inf] = -1
        data[np.isnan(data)] = 0
        labelSet = set(label)
        labelSet.remove(0)
        for l in labelSet:
            biclassifier[l] = svm.SVC()
            biLabel[l] = (label == l)
        for l in biclassifier:
            print( "training,using label %s."%str(l))
            biclassifier[l].fit(data,biLabel[l])
            print( "done")
        with open('biclassifier.svm','w') as f:
            pickle.dump(biclassifier,f)
    
    testings = getFeatureOfFace(faceMapping('./testing'))
    for k in testings:
        print( "start classifying %s."%k)
        data_k = testings[k].T[featureIndex].T
        data_k[data_k == np.inf] = 1
        data_k[data_k == -np.inf] = -1
        data_k[np.isnan(data_k)] = 0
        resTmp = {}
        shape = 0
        for l in biclassifier:
            print( "classifying,using label %s."%str(l))
            resTmp[l] = np.array(biclassifier[l].predict(data_k),dtype = np.bool)
            print( "done.")
            shape = resTmp[l].shape
        result = np.zeros(shape)
        for l in resTmp:
            result[resTmp[l]] = l
        np.savetxt("./result/%s.seg"%k,result,fmt = "%d")
        
if __name__=="__main__":
    svmSegment()
