import unittest

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from fne.genotopheno.cell_operations import OPS
from fne.genotopheno import LearnableCell, VisionNetwork
from fne.evolution import Mutations, get_conf, Crossover, get_dataset
from fne import Population

class TestSum(unittest.TestCase):
    def test_ops_shape(self):
        """
        Test that the result of the possible operations have the right shape
        """
        confs = {'C_in': 3, 'C_out': 8, 'stride': 1, 'affine': True}

        for name, layer in OPS.items():
            net = layer(**confs)
            x = torch.rand((16, confs['C_in'], 32, 32))
            y = net(x)
            self.assertEqual(list(y.shape), [16, confs['C_out'], 32, 32])
            
    def test_learnable_cell(self):
        """
        Test if a genotype is correctly encoded into a phenotype
        """
        genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
        search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                        'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}

        net = LearnableCell(3, genotype, search_space)
        x = torch.rand((16, 3, 32, 32))
        y = net(x)
        self.assertEqual(list(y.shape), [16, 7, 15, 15])

    def test_visionnetwork(self):
        """
        Test if 4 cells genotypes can be encoded into a full network
        """
        genotype1 = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
        genotype2 = '0|6|0|0|0|2|0|0  1|0|0|0|0|0|1|0  1|1|0|0|0|0|0|0--1  7'
        search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                        'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
        population = [genotype1, genotype2, genotype2, genotype1]

        net = VisionNetwork(3, 5, population, search_space, 2)
        x = torch.rand(16, 3, 129, 64)
        y = net(x)
        self.assertEqual(list(y.shape), [16, 5])

    def test_mutation(self):
        """
        Test if a cell mutates correctly
        """
        genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
        search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                        'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}

        mutator = Mutations(search_space, prob_mutation=0.8, prob_resize=0.99, prob_swap=0.99)
        mutated_g = mutator(genotype)
        mutated_g = mutator(mutated_g)
        mutated_g = mutator(mutated_g)
        a, s, d = get_conf(mutated_g)
        print('---->',mutated_g)
        self.assertGreaterEqual(10,d)
        self.assertTrue(s in (0,1))
        a = torch.tensor(a)
        d = int((a.shape[0]*2)**.5)
        start = 0
        for i in range(d):
            end = int((i+1)*(i+2)/2)
            self.assertTrue(a[start:end,:].sum()>0)
            start = end

    def test_mutation2(self):
        """
        Test update happens correctly correctly
        """
        genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
        search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                        'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}

        mutator = Mutations(search_space, prob_mutation=0.8, prob_resize=0.99, prob_swap=0.99)
        mutated_g = mutator(genotype)
        a, s, d = get_conf(mutated_g)
        mutator.update_strategy(a, True)

    def test_crossover(self):
        genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
        genotype2 = '0|0|0|3|0|1|0|0  1|1|1|1|1|1|1|1  0|1|0|9|9|0|2|1--1  7'
        search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                        'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
        crosser = Crossover(search_space, .9, .9)
        gen = crosser(genotype, genotype2)
        print('|---->',gen)
        a, s, d = get_conf(gen)
        crosser.update_strategy(a, True)

    def test_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        dataset = get_dataset(5, trainset, len(classes), distance=lambda x,y:x-y)
        self.assertTrue(len(dataset)==2000)

    def test_scoring(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        def target_transform(i):
            tmp    = torch.zeros(10)
            tmp[i] = 1.
            return tmp
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
        pop = Population(trainset)
        pop.do_one_generation()
        self.assertTrue(len(pop.population)==pop.config.pop_size)


if __name__ == '__main__':
    unittest.main()
