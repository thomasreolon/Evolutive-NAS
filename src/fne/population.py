from collections import defaultdict
import torch

from .genotopheno import LearnableCell, VisionNetwork
from .evolution import encode_conf, get_conf, get_dataset, correct_genotype, Crossover, Mutations
from .history import HistoryOfGenotypes


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
        self.C_in           = 3
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
        # load configurations
        cnf = Config(**(config or {}))
        self.config = cnf

        # set inital populaton if given, else get a random population
        if isinstance(initial_population, (list, tuple)):
            self.population = initial_population
        else:
            self.population = self.get_rand(cnf.pop_size)

        # initialize crossover and mutation classes (they will learn parameters through epochs)
        self.mutation = Mutations(cnf.search_space, cnf.mut_prob, cnf.mut_resize, cnf.mut_swap, cnf.mut_eve)
        self.crossover = Crossover(cnf.search_space, cnf.cross_prob, cnf.cross_max)

        # remember already tested architectures
        self.history = defaultdict(lambda: None)

        # store dataset
        self.dataset  = dataset

        # get info on the problem from the dataset
        inp, out      = dataset[0]
        assert len(inp.shape)==3, f"samples shape {inp.shape} is not valid, expecting (C, H, W)"
        cnf.C_in      = inp.shape[0]
        cnf.out_shape = out.shape.numel()


    def do_one_generation(self):
        cnf = self.config
        of_per_ind = self.config.offsprings // len(self.population) # offsprings x individual   

        # get offsprings and give them a score
        scores, prev_ind = [], self.population[1]
        for individual in self.population:
            for offspring in range(of_per_ind):
                offspring = self.evolve_genotype(genotype, prev_ind)
                arch = offspring.split('--')[0]
                score = self.history[arch]              # check if the architecture was already tested

                if score is None:
                    # compute fitness score for the offspring
                    score = fitness_score(offspring, cnf.C_in, cnf.search_space, self.dataset, cnf.max_distance, cnf.distance)
                    self.history[arch] = score
                scores.append((offspring, score))
            prev_ind = individual

        # get offspring with 1rank pareto dominance
        ######

        # sort them by best overall

        # get the best n for the next population



    def evolve_genotype(self, genotype, mate):
        tmp = self.mutation(genotype)
        offspring = self.crossover(tmp, mate)
        self.mutation.update_genoname(tmp, offspring)
        return offspring



    def get_rand(self, num):
        mutate = Mutations(self.config.search_space)
        geno = '0|0|0|0|0|0|0|0--1  5'
        pop = [mutate(geno) for _ in range(num)]  # random initial genotype (mutation occours with prob=self.config.mut_prob)
        return [correct_genotype(g) for g in pop] # fixes it if the mutation had no effect