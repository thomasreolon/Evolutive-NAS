from collections import defaultdict
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .genotopheno import LearnableCell, VisionNetwork
from .evolution import encode_conf, get_conf, get_dataset, correct_genotype, Crossover, Mutations, fitness_score


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

    def do_one_generation(self):
        """
        we use the first selection approach (score based) for three times
        then we use DARTS selection for the last one (darts is more expensive)
        """
        self.do_evolution_step()
        self.do_evolution_step()
        self.do_evolution_step(True)
        self.do_darts_step()

    def do_evolution_step(self, more_individuals=False):
        """ mutate the population and selects the offsprings with the best score

        --> more_individuals:   how many offsprings to keep default = number of population
                should be (True) when the consecutive step is 'do_darts_step', which selects 1 cell every tourn_size cells
        """
        cnf = self.config
        # offsprings x individual
        of_per_ind = self.config.offsprings // len(self.population)
        torch.cuda.empty_cache()

        # get offsprings and give them a score
        scores, prev_ind = [], self.population[1]
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
                scores.append((offspring, score))
            prev_ind = individual

        # get offspring with rank1 pareto dominance
        scores = self.get_pareto_rank1(scores)

        # get the best n for the next population
        # TODO: this sorting could be improved
        scores.sort(key=lambda x: x[1][2])
        n = self.config.pop_size * \
            self.config.tourn_size if more_individuals else self.config.pop_size
        self.population = [geno for geno, _ in scores[:n]]
        random.shuffle(self.population)

        # update parameters for mutation & crossover
        for geno in self.population:
            self.mutation.update_strategy(geno, True)
            self.crossover.update_strategy(geno, True)
        self.mutation.clear_cache()
        self.crossover.clear_cache()

    def do_darts_step(self):
        """
        instead of relying just on our scores. we apply DARTS mechanism to improve the selection process.
        we create a network of cells where we jointly learn the weights & an hyperparameter called alphas.
        this parameter is a matrix depthXt_size where the cells at the same depth are competing against each other to survive.
        after some epochs we get the alphas and for each depth we select the cell with the highest value. (to pass to the next generation)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t_size = self.config.tourn_size
        depth = len(self.population) // t_size
        network = VisionNetwork(
            self.config.C_in, self.config.n_classes, self.config.search_space, depth).to(device)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(network.parameters(), weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x: 1/(1+x/2)**2)
        loss_fn = nn.L1Loss()

        for _ in range(10):
            for inps, targs in loader:
                inps, targs = inps.to(device), targs.to(device)
                outs = network(inps)
                loss = loss_fn(outs, targs)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step()

        _, results = network.alphas.max(dim=1)
        pop = []
        for i in range(self.config.pop_size):
            pop.append(self.population[i*t_size+int(results[i])])
        self.population = pop

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

    def get_pareto_rank1(self, offsprings):
        goodones = []
        how_many = self.config.pop_size*self.config.tourn_size

        for offspring, (s1, s2, s3) in offsprings:
            good = True
            for _, (c1, c2, c3) in offsprings:
                if s1 > c1 and s2 > c2 and s3 > c3:
                    # found another offspring which is always better
                    # (we want to minimize the scores)
                    good = False
                    break
            if good:
                goodones.append((offspring, (s1, s2, s3)))

        if len(goodones) < self.config.pop_size*self.config.tourn_size:
            # too few offspring have rank 1 (a few dominate)
            # add some other worse offsprings
            almost_good = []
            for offspring, (s1, s2, s3) in offsprings:
                almostgood = True
                beated = 0
                for _, (c1, c2, c3) in offsprings:
                    beated += int(s1 > c1) + int(s2 > c2) + int(s3 > c3)
                if almostgood:
                    almost_good.append((beated, (offspring, (s1, s2, s3))))
            almost_good.sort(key=lambda x: x[0])
            goodones += [x[1] for x in almost_good[len(goodones):]]
        return goodones
