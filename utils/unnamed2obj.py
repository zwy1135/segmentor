# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 21:06:56 2014

@author: wy
"""

def convert(inMeshFile,outMeshFile,colorFile = None):
    with open(inMeshFile,'r') as inMesh:
        vertexString = ""
        faceString = ""
        RString = ""
        GString = ""
        BString = ""
        
        for line in inMesh:
            if line.strip().startswith("Vertex"):
                XYZRaw = line.split(":")[-1]
                XYZ = XYZRaw.strip().split(",")
                X,Y,Z = map(lambda x:float(x.split("=")[-1]),XYZ)
                vertexString += "v %f %f %f \n"%(X,Y,Z)
                
            elif line.strip().startswith("Texture"):
                RGBRaw = line.split(":")[-1]
                RGB = RGBRaw.strip().split(",")
                R,G,B = map(lambda x:int(x.split("=")[-1]),RGB)
                RString += "%d\n"%R
                GString += "%d\n"%G
                BString += "%d\n"%B
                
            elif line.strip().startswith("Triangle"):
                FSTRaw = line.split(":")[-1]
                FST = FSTRaw.strip().split(",")
                F,S,T = map(lambda x:int(x.split("=")[-1]),FST)
                faceString += "f %d %d %d \n"%(F,S,T)
            

    
    with open(outMeshFile,'w') as outMesh:
        outMesh.write(
        "#obj-file from bjut_3d \n#Vertex \n%s \n#face \n%s \n"%(vertexString,faceString)
        )
        
    if colorFile:
        with open(colorFile+".R",'w') as Rcolor:
            Rcolor.write(
            "%s"%RString
            )
        
        with open(colorFile+".G",'w') as Gcolor:
            Gcolor.write(
            "%s"%GString
            )
            
        with open(colorFile+".B",'w') as Bcolor:
            Bcolor.write(
            "%s"%BString
            )
            
if __name__ == "__main__":
    import os
    fromDir = "raw"
    toDir = "obj"
    colorDir = "color"
    fileList = os.listdir(fromDir)
    for f in fileList:
        if not f.split(".")[-1] == "txt":
            continue
        outName = f.split(".")[0]
        inMeshFile = os.path.join(fromDir,f)
        outMeshFile = os.path.join(toDir,outName + ".obj")
        colorFile = os.path.join(colorDir,outName)
        print("converting %s"%inMeshFile)
        convert(inMeshFile,outMeshFile,colorFile)
    
    
    
    
    
    