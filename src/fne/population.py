import torch

from .genotopheno import LearnableCell, VisionNetwork
from .evolution import encode_conf, get_conf, get_dataset, correct_genotype, Crossover, Mutations


class dotdict(dict):
    """dot.notation to access dictionary attributes, if no attribute returns None"""
    __getattr__ = lambda s,k: (k in s) and s[k] or None
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Config():
    """configurations for evolution"""
    def __init__(self, **kw):
        """
        Default configuration for the network

        Arguments
        ---------
        --> pop_size=5:         how many genotypes survive at each generation, each genotype is a cell
        --> offsprings=150:     how many offsprings are generated from the population size
        --> net_depth=4:        how deep the final network is (pop_size*net_depth = number of offsprings that survive to the first selection)
        --> C_in=3:             input channels, usually 3 for images [batchsize, C_in, H, W]
        --> search_space:       which type of connection to use inside the cells
        --> distance:           function that measure dissimilarity between 2 classes. distance(target1, target2)->float
        --> max_distance:       max distance mesurable between two targets
        --> mut_prob=.8:        probability to mutate a connection of a cell
        --> mut_resize=.05:     probability to add/remove a layer of the cell
        --> mut_swap=.04:       probability to swap a layer of the cell
        --> mut_eve=.5:         exploration over exploitation. a number in [0,1], lower==>more exploitation 
        --> cross_prob=.3:      probability to mix two genotypes
        --> cross_max=.2:       probability to do a crossover taking the minimum between the two parents
        """
        c = dotdict(kw)
        default_search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                    'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
        # global params
        self.pop_size       = c.pop_size        or 5
        self.offsprings     = c.offsprings      or 150
        self.net_depth      = c.net_depth       or 4
        self.C_in           = c.C_in            or 3
        self.search_space   = c.search_space    or default_search_space
        # target measures
        self.distance       = c.distance
        self.max_distance   = c.max_distance
        # mutation params
        self.mut_prob       = c.mut_prob        or .8
        self.mut_resize     = c.mut_resize      or .05
        self.mut_swap       = c.mut_swap        or .04
        self.mut_eve        = c.mut_eve         or .5
        # crossover params
        self.cross_prob     = c.cross_prob      or .3
        self.cross_max      = c.cross_max       or .2




class Population():
    """
    This class contains the genotypes of the current generation
    calling the method next_gen() evolves the current
    """
    def __init__(self, dataset, config=None, initial_population=None):
        """
        Arguments
        ---------
        dataset:                    torch.utils.Dataset
            training samples used to discover architecture, dataset[0] should return a tuple[input_tensor, target_tensor]
            with target_tensor should have the same size of the output layer of the NN

        config:                     dict
            settings of the evolutions, class Config contains the default values

        initial_population=10:      int or list of genotypes
            if an integer N is passed, a random population of N individuals is created
            if a list of genotypes is passed, it is used as the initial population

        """
        self.config = Config(**(config or {}))
        if isinstance(initial_population, (list, tuple)):
            self.population = initial_population
        else:
            self.population = self.get_rand(config.pop_size)







    def get_rand(self, num):
        mutate = Mutations(self.config.search_space)
        geno = '0|0|0|0|0|0|0|0--1  5'
        pop = [mutate(geno) for _ in range(num)]
        return [correct_genotype(g) for g in pop]