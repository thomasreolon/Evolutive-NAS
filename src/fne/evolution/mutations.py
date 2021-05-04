from collections import defaultdict
import torch
from .utils import get_conf, encode_conf


class Mutations():
    def __init__(self, search_space, prob_mutation=0.8, prob_resize=0.05, prob_swap=0.04, exploration_vs_exploitation=0.5):
        n = len(search_space)
        # general vars
        self.exploration_vs_exploitation = exploration_vs_exploitation
        self._cache = {}
        # mutations over edges
        self.sspace_used = torch.ones(n)*2
        self.sspace_success = torch.ones(n)

        # mutation edge
        self.prob_mutation = prob_mutation

        # mutation swap
        self.prob_swap = prob_swap

        # mutation reduce
        self.avg_len = 10
        self.prob_resize = prob_resize

    def __call__(self, genotype):
        """
        takes a genotype and returns a mutated one
        """
        self.exploration_vs_exploitation *= 0.99
        architecture, use_shared, dataset = get_conf(genotype)
        mutations = []

        # update architecture edges
        while(torch.rand(1)<self.prob_mutation):
            architecture, j = self.mutate_one_edge(architecture)
            mutations.append(j)
        architecture = self.mutate_swap(architecture)
        
        # update hyperparams
        r = torch.rand(2)
        if r[0]<0.04: use_shared = (use_shared+1)%2
        if r[1]<0.04: dataset = int((dataset+(r[1]-0.01)*50)%10)

        # resize architecture
        architecture, tmp = self.mutate_resize(architecture)
        mutations += tmp

        hash_arch = '.'.join([','.join([str(x) for x in ar]) for ar in architecture])
        self._cache[hash_arch] = mutations
        return encode_conf(architecture, use_shared, dataset)


    def mutate_one_edge(self, architecture):
        """basic mutation:change one edge of the cell
        how the edge is changed depends on:
        - exploitation: successfull_mutations / total_mutations  of a specific operation
        - explration:   to give more attention to low used operation
        Moreover, exploration_vs_exploitation makes exploitation a bit more important with time passing
        """
        eve = self.exploration_vs_exploitation
        rand = torch.rand(3)
        i = int(rand[0] * len(architecture))
        j = int(rand[1] * len(architecture[0]))
        #                    exploitation                                       exploration
        var = (1-eve)*(self.sspace_success[j] / self.sspace_used[j]) + eve*(1- self.sspace_used[j] / self.sspace_used.max())
        architecture[i][j]  += int((var-0.12)*rand[2]*16)
        return architecture, j

    def mutate_resize(self, architecture):
        """mutation that adds/removes layers"""
        architecture = torch.tensor(architecture, dtype=torch.int)
        n_params = torch.tensor(architecture.sum(), dtype=torch.float)
        mutations = []
        prob = n_params / (n_params+self.avg_len) * (self.prob_resize*3/4) # reduce prob   3/4 gives a bit more prob to increase rather than reduce
        prob2 = self.avg_len / (n_params+self.avg_len) * self.prob_resize  # increase prob
        depth = int((len(architecture)*2)**0.5)                            # network depth
        if len(architecture)>1 and torch.rand(1)<prob.item():
            # reduce the cell by one layer, sum the removed layers to the previous ones
            if depth > 1:
                end = int(depth*(depth-1)/2)
                for i in range(len(architecture)-end):
                    architecture[i] += ((architecture[i]+architecture[i+end])/2).int()
                    mutations.append(architecture[i+end].max(dim=0)[1].item())
                architecture = architecture[:end]
        elif torch.rand(1)<prob2.item():
            # add a new layer and apply some mutations to it
            # poss. problem:   small new layer --> bottleneck --> low scores
            n = len(architecture[0])
            new_arch = [[0]*n]*(depth+1)
            j = int(torch.rand(1)*n)
            new_arch[-1][j] += 3
            mutations.append(j)
            while(torch.rand(1)>self.prob_mutation):
                new_arch, j = self.mutate_one_edge(new_arch)
                mutations.append(j)
            architecture = torch.cat((architecture, torch.tensor(new_arch, dtype=torch.int)), dim=0)

        self.avg_len = .9*self.avg_len + .1*n_params
        return architecture.tolist(), mutations

    def mutate_swap(self, architecture):
        r = torch.rand(2)
        if r[0]<self.prob_swap and len(architecture)>1:
            tmp = architecture[0]
            i = 1+int(r[1]*(len(architecture)-1))
            architecture[0] = architecture[i]
            architecture[i] = tmp
        return architecture
    
    def update_genoname(self, old, new):
        self._cache[new] = self._cache[old]
        del self._cache[old]

    def update_strategy(self, architecture, success):
        """updates ratio successful/all_mutations for a given mutation type"""
        if isinstance(architecture, str):
            architecture, _, _ = get_conf(genotype)
        architecture = '.'.join([','.join([str(x) for x in ar]) for ar in architecture])
        for j in self._cache[architecture]:
            self.sspace_used[j] += 1
            if success:
                self.sspace_success[j] += 1
        #del self._cache[architecture]






