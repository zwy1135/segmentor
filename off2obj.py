# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 11:19:45 2014

@author: zeng
"""

#!/usr/bin/env python
#
# off2obj.py : Convert a .off mesh (Geomview) into a .obj mesh (wavefront)
#
# Copyright (C) 2010  Clement Creusot
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import sys
import os
import string


def print_help():
    print "Usage: "+os.path.basename(sys.argv[0])+" filein.off [fileout.obj]"
    sys.exit()

def print_error(str):
    print "ERROR: "+str
    sys.exit()

if (len(sys.argv) < 2):
    print_help()

offfilename = sys.argv[1];
objfilename = sys.argv[1]+".obj";
if (len(sys.argv) == 3):
    objfilename = sys.argv[2];

objfile = open(objfilename, "w")
offfile = open(offfilename, "r")

line = offfile.readline()
tab = string.split(line)
if cmp(tab[0],"OFF") != 0:
    print_error("Bad OFF file header")


line = offfile.readline()
tab = string.split(line)
while (string.find("#",tab[0]) != -1):
    line = offfile.readline()
    tab = string.split(line)

pointNb = int(tab[0])
faceNb  = int(tab[1])
edgeNb  = int(tab[2])

objfile.write("# File type: ASCII OBJ\n")

for i in range(pointNb):
    line = offfile.readline()
    t = string.split(line)
    objfile.write("v "+t[0]+" "+t[1]+" "+t[2]+"\n")



for i in range(faceNb):
    line = offfile.readline()
    t = string.split(line)
    ## In obj files the first vertex is 1 not 0
    polyDegree = int(t[0]) # number of vertex per polygone
    tOut = []        
    for j in range(polyDegree):
        tOut.append(str(int(t[j+1]) +1))
    objfile.write("f "+string.join(tOut," ")+"\n")

objfile.close()
offfile.close()