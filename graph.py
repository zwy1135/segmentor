# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:00:47 2015

@author: wy
"""

from __future__ import print_function
from itertools import combinations
from copy import copy

import numpy as np
import networkx as nx



def build_vertex_graph(faces):
    graph = nx.Graph()
    for f in faces:
        graph.add_cycle(f)
    
    return graph
    
def build_face_graph(faces):
    graph = nx.Graph()
    
    Map = {}
    for i,face in enumerate(faces):
        for v in face:
            Map.setdefault(v, set())
            Map[v].add(i)
            
    for faceSet in Map.values():
        for u,v in combinations(faceSet,2):
            intersetion = [s for s in faces[u] if s in faces[v]]
            if len(intersetion)>1:
                graph.add_edge(u,v)
                
    return graph
    
def build_weight_graph(graph,feature):
    graph2 = nx.create_empty_copy(graph)
    for edge in graph.edges_iter():
        u,v = edge
        similarity = computeSimilarity(feature[u],feature[v])
        graph2.add_edge(u,v,weight = similarity)
        
    return graph2
    
def build_s_t_graph_list(weightGraph,proba):
    graphList = []
    graphSize = weightGraph.number_of_nodes() + 2
    source = graphSize - 2
    target = graphSize - 1
    
    for label in range(1,proba.shape[1]):
        new = copy(weightGraph)
        for i,p in enumerate(proba):
            new.add_edge(source, i, weight = p[label])
            new.add_edge(i, target, weight = 1 - p[label])
            
        graphList.append(new)
        
    return graphList
    
    
def cut_and_label(graphList):
    size = graphList[0].number_of_nodes() - 2
    result = np.zeros(size,dtype=np.int8)
    source = size
    target = size + 1
    
    for i,g in enumerate(graphList):
        cutValue,partition = nx.minimum_cut(g,source,target,capacity="weight")
        _,trueIndex = partition
        
        if target in trueIndex:
            trueIndex.remove(target)
        
        result[list(trueIndex)] = i + 1
        
    return result
    
    
    
def pearson(v1,v2):
    sum1 = np.sum(v1)
    sum2 = np.sum(v2)
    
    length = len(v1)
    
    sum1Sq = np.sum(v1**2)
    sum2Sq = np.sum(v2**2)
    
    pSum = np.sum(v1*v2)
    
    num = pSum - (sum1*sum2/length)
    den = np.sqrt((sum1Sq - sum1**2/length)*(sum2Sq - sum2**2/length))
    if den == 0:
        return np.inf
    
    return num/den

def computeSimilarity(feature1,feature2):
    assert len(feature1)==len(feature2)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    similarity = pearson(feature1,feature2)
    
    return similarity
    
            
    
    
    

