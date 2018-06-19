from ase.cluster.bcn_factory import ClusterFactory

class CutClusterFactory(ClusterFactory):
    ##print('Call to CutClusterFactory')
    
    cl_cut = True
    
    ##print('Cut cluster factory calculation')
     ##def activate_cut(self):
        ##elf.cl_cut = True
    
  
    ##print('cl_cut, CutClusterFactory', cl_cut)
    
CutCluster = CutClusterFactory()
    


    
