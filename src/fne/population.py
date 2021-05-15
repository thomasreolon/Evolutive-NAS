from collections import defaultdict
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .genotopheno import LearnableCell, VisionNetwork
from .evolution import get_dataset, correct_genotype, Crossover, Mutations, fitness_score
from .evolution.utils import clear_cache, get_memory

from .genotopheno import cells

class dotdict(dict):
    """dot.notation to access dictionary attributes, if no attribute returns None"""
    def __getattr__(s, k): return (k in s) and s[k] or None
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Config():
    """configurations for evolution"""

    def __init__(self, dictionary=None):
        """
        Default configuration for the network

        Arguments
        ---------
        --> pop_size=5:         how many genotypes survive at each generation, each genotype is a cell
        --> offsprings=150:     how many offsprings are generated from the population size
        --> tourn_size=4:       how many contendents a cell will fight with in the darts step (pop_size*tourn_size = number of offsprings that survive to the first selection)
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
        if isinstance(dictionary, Config):
            dictionary = dictionary.__dict__
        c = dotdict(dictionary or {})
        default_search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                                'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
        # global params
        self.pop_size = c.pop_size or 5
        self.offsprings = c.offsprings or 150
        self.tourn_size = c.tourn_size or 4
        self.C_in = 3
        self.search_space = c.search_space or default_search_space
        # target measures
        self.distance = c.distance
        self.max_distance = c.max_distance
        # mutation params
        self.mut_prob = c.mut_prob or .8
        self.mut_resize = c.mut_resize or .05
        self.mut_swap = c.mut_swap or .04
        self.mut_eve = c.mut_eve or .5
        # crossover params
        self.cross_prob = c.cross_prob or .3
        self.cross_max = c.cross_max or .2


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
        cnf = Config((config or {}))
        self.config = cnf

        # set inital populaton if given, else get a random population
        if isinstance(initial_population, (list, tuple)):
            self.population = initial_population
        else:
            self.population = self.get_rand(cnf.pop_size)
        self.best_offspring = self.population[0]

        # initialize crossover and mutation classes (they will learn parameters through epochs)
        self.mutation = Mutations(
            cnf.search_space, cnf.mut_prob, cnf.mut_resize, cnf.mut_swap, cnf.mut_eve)
        self.crossover = Crossover(
            cnf.search_space, cnf.cross_prob, cnf.cross_max)

        # remember already tested architectures
        self.history = defaultdict(lambda: None)

        # store dataset
        self.dataset = dataset

        # get info on the problem from the dataset
        inp, out = dataset[0]
        assert len(
            inp.shape) == 3, f"samples shape {inp.shape} is not valid, expecting (C, H, W)"
        cnf.C_in = inp.shape[0]
        cnf.n_classes = out.numel()

    def do_one_generation(self, verbose=False):
        """
        we use the first selection approach (score based) for three times
        then we use DARTS selection for the last one (darts is more expensive)
        """
        self.do_evolution_step(verbose=verbose)
        self.do_evolution_step(verbose=verbose)
        self.do_evolution_step(True, verbose=verbose)
        self.do_darts_step(verbose=verbose)

    def do_evolution_step(self, more_individuals=False, verbose=False):
        """ mutate the population and selects the offsprings with the best score

        --> more_individuals:   how many offsprings to keep default = number of population
                should be (True) when the consecutive step is 'do_darts_step', which selects 1 cell every tourn_size cells
        """
        if verbose: print('EVOLUTION STEP BEGIN')
        cnf = self.config
        # offsprings x individual
        of_per_ind = self.config.offsprings // len(self.population)
        # how many offsprings to keep
        n = self.config.pop_size * self.config.tourn_size \
            if more_individuals else self.config.pop_size

        # get offsprings and give them a score
        scores, prev_ind = {}, self.population[1]
        toprint = 1e9
        while len(scores)<n:
            for individual in self.population:
                for offspring in range(of_per_ind):
                    offspring = self.evolve_genotype(individual, prev_ind)
                    arch = offspring.split('--')[0]
                    # check if the architecture was already tested
                    score = self.history[arch]

                    if score is None:
                        # compute fitness score for the offspring
                        score = fitness_score(
                            offspring, cnf.C_in, cnf.search_space, self.dataset, cnf.max_distance, cnf.distance)
                        self.history[arch] = score
                    
                    # no identical genotypes reinserted
                    scores[offspring] = score
                    if verbose and score[2]<toprint: print('[',score,'] -> score offspring', offspring) ; toprint = score[2]
                prev_ind = individual

        # get offspring with rank1 pareto dominance
        scores = list(scores.items())
        scores = self.get_pareto_rank1(scores, n)

        # get the best n for the next population
        scores = self.sort_scores_by_rank_sum(scores)

        # select n offsprings
        self.population = [geno for geno, _ in scores[:n]]
        self.best_offspring = self.population[0]
        random.shuffle(self.population)

        # update parameters for mutation & crossover
        for geno in self.population:
            self.mutation.update_strat_good(geno)
            self.crossover.update_strat_good(geno)
        self.mutation.update_strat_bad()
        self.crossover.update_strat_bad()
        if verbose: print('EVOLUTION STEP END')

    def sort_scores_by_rank_sum(self, scores):
        tot = {s:0 for s in scores}
        # score = sum of the rank
        # the lower the better
        for k in range(3):
            scores.sort(key=lambda x: x[1][k])
            for i,s in enumerate(scores):
                tot[s] += i
        tot = list(tot.items())
        # sort by rank
        tot.sort(key=lambda x: x[1])
        return [s for s,_ in tot]


    def do_darts_step(self, verbose=False):
        """
        instead of relying just on our scores. we apply DARTS mechanism to improve the selection process.
        we create a network of cells where we jointly learn the weights & an hyperparameter called alphas.
        this parameter is a matrix depthXt_size where the cells at the same depth are competing against each other to survive.
        after some epochs we get the alphas and for each depth we select the cell with the highest value. (to pass to the next generation)
        """
        if verbose: print('DARTS STEP BEGIN')

        # network settings
        t_size = self.config.tourn_size
        depth = len(self.population) // t_size        # = self.config.pop_size
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # GPU memory info
        network = VisionNetwork(
            self.config.C_in, self.config.n_classes, self.population, self.config.search_space, depth)
        n_params = sum([p.numel() for p in network.parameters()])
        w = self.dataset[0][0].shape[2]
        free_mem  = torch.cuda.get_device_properties(0).total_memory -torch.cuda.memory_allocated(0)
        batch_size = max(1, int(free_mem /(n_params * (w/5)**2 *4)))   # memory used by NN_activations-> nparams*ratioWidth/Kernel*sizeofFloat
        network = network.to(device)
        if verbose: print(f'n params = {n_params}, batch_size={batch_size}')
        
        # briefly train network to get improved weights (not modifying alphas)
        network.alphas.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=.004, weight_decay=1e-5)
        loss_fn = nn.L1Loss()
        prev_loss=None
        for e in range(9):
            tot_loss,cc = 0, 0
            # whole trainingset can be too expensive
            small_dataset = get_dataset(e, self.dataset, self.config.max_distance, self.config.distance)
            loader = DataLoader(small_dataset, batch_size, shuffle=True)
            # backpropagation to learn the aplhas (and some weights)
            for inps, targs in loader:
                clear_cache()
                inps, targs = inps.to(device), targs.to(device)
                outs = network(inps)
                loss = loss_fn(outs, targs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, network.parameters()), 2.)
                optimizer.step()
                optimizer.zero_grad()
                # stats on how training is doing
                if prev_loss is None: prev_loss = loss.item()/inps.shape[0]
                tot_loss+=loss.item()
                cc += inps.shape[0]
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*(.5 + float(tot_loss/cc < prev_loss)*2.5)
            prev_loss = tot_loss/cc

        # whole trainingset can be too expensive
        network.alphas.requires_grad = True
        optimizer = torch.optim.Adam(network.parameters(), lr=.004, weight_decay=1e-5)
        small_dataset = get_dataset(9, self.dataset, self.config.max_distance, self.config.distance)
        loader = DataLoader(small_dataset, batch_size, shuffle=True)
        # backpropagation to learn the aplhas (and some weights)
        for inps, targs in loader:
            clear_cache()
            inps, targs = inps.to(device), targs.to(device)
            outs = network(inps)
            loss = loss_fn(outs, targs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 2.)
            optimizer.step()
            optimizer.zero_grad()
        if verbose: print('alphas:',network.alphas.tolist())

        # lower alphas --> high -log_softmax --> more inportance in the network
        _, results = network.alphas.min(dim=1)
        pop = []
        for i in range(self.config.pop_size):
            pop.append(self.population[i*t_size+int(results[i])])
        self.population = pop
        if verbose: print('DARTS STEP END')

    def evolve_genotype(self, genotype, mate):
        # perform mutation with prob a
        tmp = self.mutation(genotype)
        # perform crossover with prob b
        tmp2 = self.crossover(tmp, mate)
        # correct genotype if it has errors (eg. all zeros)
        offspring = correct_genotype(tmp2)
        # update the genotype name in the cache
        self.mutation.update_genoname(tmp, offspring)
        # update the genotype name in the cache
        self.crossover.update_genoname(tmp2, offspring)
        return offspring

    def get_rand(self, num):
        mutate = Mutations(self.config.search_space)
        geno = '0|0|0|0|0|0|0|0--1  5'
        # random initial genotype (mutation occours with prob=self.config.mut_prob)
        pop = [mutate(geno) for _ in range(num)]
        # fixes it if the mutation had no effect
        return [correct_genotype(g) for g in pop]

    def get_pareto_rank1(self, offsprings, n):
        goodones = {}

        for offspring, (s1, s2, s3) in offsprings:
            good = True
            for _, (c1, c2, c3) in offsprings:
                if s1 > c1 and s2 > c2 and s3 > c3:
                    # found a contendant offspring which is always better that s
                    # (we want to minimize the scores)
                    good = False
                    break
            if good:
                goodones[offspring] = (s1, s2, s3)

        almost_good, fill = [], 1
        if len(goodones) < n:
            # too few offspring have rank 1 (a few dominate)
            # add some other worse offsprings
            fill = n-len(goodones)
            for offspring, (s1, s2, s3) in offsprings:
                if offspring in goodones: continue          # already passed selection
                almostgood = True
                beated = 0
                for _, (c1, c2, c3) in offsprings:
                    beated += int(s1 > c1) + int(s2 > c2) + int(s3 > c3)
                if almostgood:
                    almost_good.append((beated, (offspring, (s1, s2, s3))))
            almost_good.sort(key=lambda x: x[0])
        
        # precedence to rank 1, if they are not enough: fill with less beated
        goodones = list(goodones.items())
        filler = [x[1] for x in almost_good[:fill]]
        return goodones + filler
