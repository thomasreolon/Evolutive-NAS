import torch
import random
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
        self.exploration_vs_exploitation *= 0.996
        architecture, use_shared, dataset = get_conf(genotype)
        mutations = []

        # azzerate smallest entry at level 0
        if torch.rand(1)<self.prob_mutation:
            k, min_, cc= 0, 9999, 0
            for i, val in enumerate(architecture[0]):
                if val>0: cc+=1
                if val<min_ and val>0: k, min_= i, val
            if cc> len(architecture[0])/2:
                architecture[0][k] = 0

        # update architecture edges
        if (torch.rand(1)<self.prob_mutation):
            architecture, j = self.mutate_one_edge(architecture)
            mutations.append(j)
        architecture = self.mutate_swap(architecture)
        
        # update hyperparams
        r = torch.rand(2)
        if r[0]<0.04: use_shared = (use_shared+1)%2
        if r[1]<0.24: dataset = int((dataset+(r[1]-0.01)*50)%10)

        # resize architecture
        if (torch.rand(1)<self.prob_mutation):
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
        #                           exploitation                                       exploration
        weights = [(1-eve)*(self.sspace_success[j] / self.sspace_used[j]) + eve*(1- self.sspace_used[j] / self.sspace_used.max())+.2 for j in range(len(self.sspace_used))]

        j = random.choices(list(range(len(architecture[0]))) , weights=weights, k=1)[0]

        if rand[0]<0.5:
            i = random.randint(0, len(architecture)-1)       # random node
        else:
            i = random.randint(0, int((len(architecture)*2)**0.5))
            i = (i*(i+1)//2)-1                               # backbone path

        for k, block in enumerate(architecture):
            if block[j]>0 and torch.rand(1)>.3:
                i = k
        
        architecture[i][j]  = max(architecture[i][j], 2**int(rand[2]*6))
        return architecture, j

    def mutate_resize(self, architecture):
        """mutation that adds/removes layers"""
        architecture = torch.tensor(architecture, dtype=torch.int)
        n_params = float(architecture.sum())
        mutations = []
        prob_reduce = n_params / (n_params+self.avg_len) * (self.prob_resize*3/4)   # reduce prob   3/4 gives a bit more prob to increase rather than reduce
        prob_increase = self.avg_len / (n_params*(3/4)+self.avg_len) * self.prob_resize  # increase prob
        prob_reduce -= 4**len(architecture) / 4**5
        depth = int((len(architecture)*2)**0.5)                              # network depth
        if len(architecture)>1 and torch.rand(1)<prob_reduce:
            # reduce the cell by one layer, sum the removed layers to the previous ones
            if depth > 1:
                end = int(depth*(depth-1)/2)
                for i in range(len(architecture)-end):
                    if (torch.rand(1)<.3):
                        architecture[i] += ((architecture[i]+architecture[i+end])/2).int()
                        mutations.append(architecture[i+end].max(dim=0)[1].item())
                architecture = architecture[:end]
        elif torch.rand(1)<prob_increase:
            # add a new layer and apply some mutations to it
            # poss. problem:   small new layer --> bottleneck --> low scores
            n = len(architecture[0])
            new_arch = [[0]*n for _ in range(depth+1)]
            j = int(torch.rand(1)*n)
            new_arch[-1][j] += 16
            mutations.append(j)
            architecture = torch.cat((architecture, torch.tensor(new_arch, dtype=torch.int)), dim=0)
        n_params = float(architecture.sum())

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
        """updates ratio successful/all_mutations for a given mutation type"""
        if isinstance(architecture, str):
            architecture, _, _ = get_conf(architecture)
        architecture = '.'.join([','.join([str(x) for x in ar]) for ar in architecture])
        if architecture not in self._cache: return
        for j in self._cache[architecture]:
            self.sspace_used[j] += 1
            self.sspace_success[j] += 1
        del self._cache[architecture]

    def update_strat_bad(self):
        for _,v in self._cache.items():
            self.sspace_used[v] += 1
        self._cache = {}





