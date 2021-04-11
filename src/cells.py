from collections import defaultdict
import torch
import torch.nn as nn

from cell_operations import OPS


_cache = 

class FullCell(nn.Module):
    def __init__(self, C_in, C_out, genotype, search_space):
        search_space = {'dil_conv_3x3','dil_conv_5x5','dil_conv_7x7','skip_connect','clinc_3x3':  ,'clinc_7x7':  ,'avg_pool_3x3',  'max_pool_3x3'}
        
        ops = seld._get_ops(OPS, search_space)
        self.layers   = nn.ModuleList()

    def _get_ops(self, OPS, to_use):
        return {k:v for k,v in OPS.items() if k in to_use}

class LearnableCell(nn.Module):
    def __init__(self, C_in, C_out, genotype, search_space, shared_weights):
        super().__init__()
        genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7  1|1|1|1|1|1|1|1'
        search_space = {'dil_conv_3x3','dil_conv_5x5','dil_conv_7x7','skip_connect','clinc_3x3':  ,'clinc_7x7':  ,'avg_pool_3x3',  'max_pool_3x3'}

        # translate genotype
        architecture, use_shared, _, stds = self._get_conf(genotype)

        # check if genotype can be coded into phenotype
        assert len(architecture[0]) == len(search_space), 'to code genotype into phenotype, search space lenght must be equal to genotype\'s'
        assert int((len(architecture)*2)**0.5)*int((len(architecture)*2)**0.5+1)/2 == len(architecture), 'num of connection should be = n*(n+1)/2'

        # net_architecture:   layer_i reads from node_in_i
        self.genotype = genotype
        self.layers   = None
        self.node_in  = []


        # build net
        if use_shared:
            # share params with parent net
            self.layers = shared_weights
        else:
            # initialize new parameters
            self.layers = nn.ModuleList()
            for i, conn in enumerate(arch.split('-')):

    def _get_conf(self, genotype):
        architecture, evol_strattegy = genotype.split('--')
        architecture = [[int(x) for x in conn.split('|')]  for conn in architecture.split('  ')]

        use_shared, dataset, stds = evol_strattegy.split('  ')
        use_shared, dataset, stds = int(use_shared), int(dataset), [float(x) for x in stds.split('|')] 
        return architecture, use_shared, dataset, stds
    
    def get_dataset_n(self):
        return int(self.genotype.split('--')[1].split('  ')[1])



    



