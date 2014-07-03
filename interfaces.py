
import os


appPath = '~/ms/MeshsegBenchmark-1.0/exe'
descriptorPath = '~/ms/descriptormap/bin'

groupSegEval = appPath + '/groupSegEval'
iseg = appPath + '/iseg'
msh2msh = appPath + '/msh2msh'
mshinfo = appPath + '/mshinfo'
mshview = appPath + '/mshview'
outputSegments = appPath + '/outputSegments'
segAnalysis = appPath + '/segAnalysis'
segEval = appPath + '/segEval'
segPreprocess = appPath + 'segPreprocess'

generateMaps = descriptorPath + '/generateMaps'

def convert(meshIn,meshOut,option=''):
    '''
    To convert mesh files between difference formats ("ply", "off", "obj")
  Usage: msh2msh fnMeshIn fnMeshOut [options]
    '''
    usage = '%s %s %s %s'%(msh2msh,meshIn,meshOut,option)
    print 'doing:%s'%usage
    os.system(usage)
    
def view(mesh,seg):
    '''
     To view a mesh 
     Usage: mshview fnMesh [-seg fnSeg] [options]
    '''
    usage = mshview+' '+mesh+' '+seg
    print 'doing:%s'%usage
    os.system(usage)
    
def mapping(fileIn,
            config = descriptorPath + '/../examples/configuration/config.conf'
            ,field = 'c0-g-k1,c0-g-k2,H0,K0,SI0,lC0,vol0,DLP0'):
    '''
    Usage : ./generateMaps config.conf file.obj [fieldName]
    Options : 
        -i : interactive mode

    '''
    usage = '%s %s %s %s'%(generateMaps,config,fileIn,field)
    print 'doing : %s'%usage
    os.system(usage)



