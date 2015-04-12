# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:27:47 2015

@author: wy
"""
from __future__ import print_function

import numpy as np

from IO import load_model, load_feature, load_labels
from graph import build_vertex_graph, build_face_graph

dataPath = "./objdata"
testPath = "./testing"
featurePath = "./feature"
labelPath = "./seg"


class Model:

    """
    The 3-d model and it's data and graph
    """

    def __init__(self, name, isTraining=False):
        self._name = name
        self._isTraining = isTraining
        self._vertexes = None
        self._faces = None
        self._feature = None
        self._vertexGraph = None
        self._faceGraph = None
        self._labels = None

    @property
    def name(self):
        return self._name
        
    @property
    def isTraining(self):
        return self._isTraining

    @property
    def vertexes(self):
        if self._vertexes is None:
            print("loading model of %s"%self.name)
            self.get_model()
        return self._vertexes

    @property
    def faces(self):
        if self._faces is None:
            print("loading model of %s"%self.name)
            self.get_model()
        return self._faces

    @property
    def feature(self):
        if self._feature is None:
            print("loading features of %s"%self.name)
            self.get_feature()
        return self._feature

    @property
    def vertexGraph(self):
        if self._vertexGraph is None:
            print("creating vertex graph of %s"%self.name)
            self.get_vertex_graph()
        return self._vertexGraph

    @property
    def faceGraph(self):
        if self._faceGraph is None:
            print("creating face graph of %s"%self.name)
            self.get_face_graph()
        return self._faceGraph

    @property
    def labels(self):
        if self._labels is None and self.isTraining:
            print("getting labels of %s"%self.name)
            self.get_labels()
        return self._labels

    #######################
    # Working functions from here.
    #######################

    def get_model(self):
        if self._isTraining:
            path = dataPath
        else:
            path = testPath
        
        self._vertexes,self._faces = load_model(path,self.name)
        
        return self._vertexes,self._faces
        
    def get_feature(self):
        path = featurePath
        
        feature = load_feature(path, self.name)
        
        #add x,y,z to feature
        vertex_feature = np.zeros((feature.shape[0],feature.shape[1]+3))
        vertex_feature[:,:3] = self.vertexes
        vertex_feature[:,3:] = feature
        
        self._feature = [np.average(vertex_feature[face], axis=0) for face in self.faces]
        self._feature = np.array(self._feature)
        
        return self._feature
        
    def get_vertex_graph(self):
        self._vertexGraph = build_vertex_graph(self.faces)
        
        return self._vertexGraph
        
    def get_face_graph(self):
        self._faceGraph = build_face_graph(self.faces)
        
        return self._faceGraph
        
    def get_labels(self):
        path = labelPath
        self._labels = load_labels(path, self.name)
        
        
if __name__=="__main__":
    name = "F0001A1EnCtrimR0"
    m = Model(name,True)
    print(m.name)
    print(m.isTraining)
    print(m.feature)
    print(m.vertexes)
    print(m.faces)
    print(m.vertexGraph.nodes())
    print(m.faceGraph.nodes())
    print(m.labels)
    #All right
        
        

