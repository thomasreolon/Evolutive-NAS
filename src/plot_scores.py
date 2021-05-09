###############
#  we create some: random architectures, famous architectures and handcrafted architectures
#  we then use give them a score and plot the results
#
#  we expect to see that random architectures have bad scores while famous architectures have goos scores
###############

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import sys, gc

from fne.genotopheno import LearnableCell, VisionNetwork
from fne.evolution import Mutations
from fne.evolution.fitness import score_NTK, score_linear, n_params
from fne.evolution.utils import print_memory, print_, clear_cache

# dataset

print_('loading dataset:         ')

transform = transforms.Compose(
    [transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, 8,  True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# random networks


def get_random_net(n_classes):
    """
    get a NN from a randomly mutated genotype
    """
    search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                    'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
    genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
    mutator = Mutations(search_space, prob_mutation=0.8,
                        prob_resize=0.4, prob_swap=0.6)
    for i in range(6):
        genotype = mutator(genotype)
    """
    net = nn.Sequential(
        LearnableCell(3, genotype, search_space),
        nn.AdaptiveAvgPool2d(1),
        nn.Linear(c_out, n_classes)
    )
    """
    net = VisionNetwork(3, n_classes, [genotype], search_space, 1)
    return net

# famous networks


print_('defining architectures:  ')


def get_alexnet(n_classes):
    alexnet = models.alexnet()
    alexnet.classifier[6] = nn.Linear(4096, n_classes)
    return alexnet


def get_resnet(n_classes):
    resnet = models.resnet18()
    resnet.fc = nn.Linear(512, n_classes)
    return resnet


def get_vgg(n_classes):
    model = models.vgg16()
    model.classifier[6] = nn.Linear(4096, n_classes)
    return model


def get_densenet(n_classes):
    model = models.densenet161()
    model.classifier = nn.Linear(4096, n_classes)
    return model


# hancrafted networks
def net1(n_classes, C_in=3):
    return nn.Sequential(nn.Conv2d(C_in, 64, 5), nn.ReLU(inplace=True), nn.Conv2d(64, 256, 3, stride=2), nn.ReLU(inplace=True), nn.AdaptiveMaxPool2d(1), nn.Linear(256, n_classes))


def net2(n_classes):
    end = net1(n_classes, 8)
    return nn.Sequential(nn.Conv2d(3, 8, 11), nn.BatchNorm2d(8), nn.ReLU(inplace=True), nn.Dropout2d(0.2), end)


def net3(n_classes):
    return nn.Sequential(nn.Conv2d(3, 32, 11), nn.ReLU(inplace=True), nn.AdaptiveMaxPool2d(5), nn.Dropout(.3), nn.Linear(32*5**2, 128), nn.ReLU(inplace=True), nn.Linear(128, n_classes), nn.Softmax())


print('Done')

# get score models

def get_scores(model):
    s1 = score_NTK(loader, model, device, 200)
    s2 = score_linear(loader, model, device, 200)
    s3 = s1*s2*n_params(model)
    return s1, s2, s3


# test models

print_('getting scores:          0% ')
n_classes = 10

rand_scores = []
for _ in range(1):
    clear_cache()
    model = get_random_net(n_classes)
    model = model.to(device)
    rand_scores.append(get_scores(model))

exit(0)

print_('60% ')
famous_scores = []
for getter in [get_alexnet, get_resnet, get_vgg, get_densenet]:
    clear_cache()
    model = getter(n_classes)
    model = model.to(device)
    famous_scores.append(get_scores(model))

print_('90% ')
hand_scores = []
for getter in [net1, net2, net3]:
    clear_cache()
    model = getter(n_classes)
    model = model.to(device)
    hand_scores.append(get_scores(model))

print('100%')
scores = [rand_scores, famous_scores, hand_scores]
with open('results.txt', 'w') as fout:
    import json
    json.dump(scores, fout)

# plot first 2 scores

colors = ['r', 'g', 'b']
labels = ['random architecture',
          'popular architecture', 'average architecture']
print(scores)

for s, c, l in zip(scores, colors, labels):
    x = [v[0] for v in s]
    y = [v[1] for v in s]
    plt.scatter(x, y, color=c, label=l)

plt.title('scores for different architectures: better closer to origin')
plt.xlabel('Neural Tangent Kernel Score')
plt.ylabel('Linear Region Score')
plt.legend()
plt.show()

# plot 3° feat
count = 0
for s, c, l in zip(scores, colors, labels):
    x = [count+i for i in range(len(s))]
    y = [v[2] for v in s]
    plt.scatter(x, y, color=c, label=l)
    count += len(scores)

plt.title('relation between architectures and the third score')
plt.ylabel('score1*score2*n_params')
plt.legend()
plt.show()

