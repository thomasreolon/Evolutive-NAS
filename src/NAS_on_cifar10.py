import torch
import torchvision
import torchvision.transforms as transforms

from fne.genotopheno.cell_operations import OPS
from fne import Population, Config

import os, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))



settings = Config()
settings.max_distance = 10
settings.offsprings = 20
settings.pop_size = 6
settings.tourn_size = 2
settings.mut_resize = .55
settings.mut_swap = .4
settings.search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                    'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}

#### setting up dataset
def dist(a,b):
    _, i = a.max(dim=0)
    _, j = b.max(dim=0)
    return torch.abs(i-j)
settings.distance = dist

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def target_transform(i):
    tmp    = torch.zeros(10)
    tmp[i] = 1.
    return tmp

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)


if 'cifar_results.txt' in os.listdir('.'):
    with open('cifar_results.txt', 'r') as fin:
        init_pop = json.load(fin)['populations'][-1]
else:
    init_pop=[0]*3
    init_pop[0] = '8|0|0|0|0|0|0  0|0|0|16|0|0|0  0|16|16|0|0|0|32  0|0|0|0|0|0|0   32|0|0|0|0|0|0  32|0|0|0|0|0|0  0|0|0|0|0|0|0  0|0|0|0|0|0|0   0|0|0|0|0|0|64  0|0|0|0|64|0|64--1  7'
    init_pop[1] = '32|0|0|0|0|0|0  0|0|0|0|0|0|0  0|64|0|0|0|0|0--1  4'
    init_pop[2] = '0|8|0|0|0|4|4  0|0|0|0|2|0|0  16|8|16|8|0|0|0--1  1'


pop = Population(trainset, settings, initial_population=init_pop)

best, populations, scores = [], [], []
for i in range(12):
    if i%4==2:
        pop.do_evolution_step(True, True)
    elif i%4==3:
        pop.do_darts_step(True)
    else:
        pop.do_evolution_step(False, True)

    
    populations.append(pop.population)
    best.append(pop.best_offspring)
    scores.append(pop.scores)

    with open('cifar_results.txt', 'w') as fout:
        res={'best_per_epoch':best, 'populations':populations, 'scores':scores}
        json.dump(res, fout, indent=2)

