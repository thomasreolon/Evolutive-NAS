from collections import defaultdict
import torch
import torch.nn as nn

from .cell_operations import OPS


# TODO: add param --> parent complexity ((the cost of adding parameters falls on the descendents))

class LearnableCell(nn.Module):
    def __init__(self, C_in, genotype, search_space, do_pool=None):
        super().__init__()

        # translate genotype
        architecture, use_shared, _ = self._get_conf(genotype)

        # check if genotype can be coded into phenotype
        assert len(architecture[0]) == len(search_space)
        depth = int((len(architecture)*2)**0.5)
        assert depth * (depth+1) / 2 == len(architecture)

        # net_architecture:   layer_i reads from node_in_i
        self.genotype = genotype
        self.depth = depth
        self.layers = nn.ModuleList()
        self.node_in = []
        self.node_out = []
        self.size_in = [0 for _ in range(depth+1)]
        self.size_in[0] = C_in

        self.do_pool = do_pool
        self.pooling = nn.AvgPool2d((3, 3), stride=2)

        # build net
        for i in range(1, depth+1):
            for j in range(i):
                for op, c_out in zip(OPS.keys(), architecture[int(i*(i-1)/2+j)]):
                    if c_out > 0:
                        c_in = self.size_in[j]
                        #  connection = {op} {node_j}-->{node_i}  :  {c_in}-->{c_out}
                        layer = OPS[op](c_in, c_out, 1, True)
                        self.layers.append(layer)
                        self.node_in.append(j)
                        self.node_out.append(i)
                        self.size_in[i] += c_out

        self.C_out = self.size_in[-1]

    def _get_conf(self, genotype):
        architecture, evol_strategy = genotype.split('--')
        architecture = [[int(x) for x in conn.split('|')]
                        for conn in architecture.split('  ')]

        use_shared, dataset = evol_strategy.split('  ')
        use_shared, dataset = int(use_shared), int(dataset)
        return architecture, use_shared, dataset

    def get_dataset_n(self):
        return int(self.genotype.split('--')[1].split('  ')[1])

    def forward(self, x):
        inputs = [[] for _ in range(self.depth+1)]
        inputs[0] = x
        current = 1
        for l_in, l_out, layer in zip(self.node_in, self.node_out, self.layers):
            if l_out != current:
                inputs[current] = torch.cat(inputs[current], dim=1)
                current = l_out
            res = layer(inputs[l_in])
            inputs[l_out].append(res)

        x = torch.cat(inputs[-1], dim=1)
        img_size = min(x.shape[2], x.shape[3])
        if self.do_pool or (self.do_pool is None and img_size>50):
            ## do pooling if image is still high resolution OR specified
            self.do_pool = True
            x = self.pooling(x)
        return x
