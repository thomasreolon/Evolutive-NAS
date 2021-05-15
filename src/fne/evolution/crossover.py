import torch
from .utils import get_conf, encode_conf

"""


"""


class Crossover():

    def __init__(self, search_space, prob_crossover=0.3, prob_cross_max=0.2):
        # modified during iterations
        self.prob_crossover = prob_crossover
        self.prob_cross_max = prob_cross_max

        # block len
        self.n = len(search_space)

        # remember what type of crossover was done when updating probs
        self._cache = {}

    def __call__(self, genotype1, genotype2):
        """
        takes 2 genotypes and returns a mixed one
        """

        architecture1, use_shared1, dataset1 = get_conf(genotype1)
        architecture2, use_shared2, dataset2 = get_conf(genotype2)

        r = torch.rand(1)
        p = (1-self.prob_crossover)*(1-self.prob_cross_max)
        p1 = self.prob_crossover/(self.prob_crossover+self.prob_cross_max)
        p2 = self.prob_cross_max/(self.prob_crossover+self.prob_cross_max) + p1
        if r < p1*(1-p):
            if len(architecture1)>len(architecture2):
                architecture1 = self.crossover(architecture1, architecture2)
            else:
                architecture1 = self.crossover(architecture2, architecture1)
            hash_arch = '.'.join([','.join([str(x) for x in ar]) for ar in architecture1])
            self._cache[hash_arch] = 0
        elif r < p2*(1-p):
            architecture1 = self.max_crossover(architecture1, architecture2)
            hash_arch = '.'.join([','.join([str(x) for x in ar]) for ar in architecture1])
            self._cache[hash_arch] = 1

        if torch.rand(1) < 0.5: use_shared1 = use_shared2
        if torch.rand(1) < 0.5: dataset1 = dataset2

        genotype = encode_conf(architecture1, use_shared1, dataset1)
        return genotype

    def crossover(self, arch1, arch2):
        """inherit randomly from 1 or 2"""
        for b1,b2 in zip(arch1, arch2):
            for i in range(self.n):
                if torch.rand(1)>0.5:
                    b1[i] = b2[i]
        length = len(arch1) if torch.rand(1)>0.5 else len(arch2)
        return arch1[:length]

    def max_crossover(self, arch1, arch2):
        """inherit deterministically from smallest --> reduce complexity"""
        for b1,b2 in zip(arch1, arch2):
            for i in range(self.n):
                b1[i] = min(b1[i], b2[i])
        length = len(arch1) if len(arch1)<=len(arch2) else len(arch2)
        return arch1[:length]

    def update_genoname(self, old, new):
        if old==new: return
        if isinstance(old, str):
            old, _, _ = get_conf(old)
        old = '.'.join([','.join([str(x) for x in ar]) for ar in old])
        if old not in self._cache: return
        if isinstance(new, str):
            new, _, _ = get_conf(new)
        new = '.'.join([','.join([str(x) for x in ar]) for ar in new])
        self._cache[new] = self._cache[old]
        del self._cache[old]


    def update_strat_good(self, architecture):
        """which crossing strategy is better"""
        if isinstance(architecture, str):
            architecture, _, _ = get_conf(architecture)
        architecture = '.'.join([','.join([str(x) for x in ar]) for ar in architecture])
        if architecture not in self._cache: return
        if self._cache[architecture]==0:
            self.prob_crossover = self.prob_crossover*0.98 +0.02
        else:
            self.prob_cross_max = self.prob_cross_max*0.98 +0.02
        del self._cache[architecture]

    def update_strat_bad(self):
        """assume all the others failed"""
        for _,v in self._cache.items():
            if v==0: self.prob_crossover = self.prob_crossover*0.98
            else: self.prob_cross_max = self.prob_cross_max*0.98
        self._cache = {}

