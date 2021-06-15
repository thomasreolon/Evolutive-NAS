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
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json, os

from fne.evolution import Mutations
from fne.evolution.fitness import score_NTK, score_jacob, n_params, score_activations
from fne.evolution.utils import print_, clear_cache, correct_genotype

import os

from fne.genotopheno.network import EvaluationNetwork
os.chdir(os.path.dirname(os.path.realpath(__file__)))

### how many time test the same model
repeat = 4

# dataset

print_('loading dataset:         ')

transform = transforms.Compose(
    [transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# random networks


def get_random_net(n_classes):
    """
    get a NN from a randomly mutated genotype
    """
    search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                    'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
    nparam = 1e6
    while (nparam>=1e6):
        genotype = '0|0|2|0|0|2|0  1|0|0|1|1|0|0  0|1|0|0|0|0|2--1  7'
        mutator = Mutations(search_space, prob_mutation=0.4,
                            prob_resize=0.3, prob_swap=0.4)
        for i in range(6):
            genotype = mutator(genotype)
        genotype = correct_genotype(genotype)
        net = EvaluationNetwork(3, n_classes, [genotype], search_space)
        nparam = sum(p.numel() for p in net.parameters())
    clear_cache()
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
    return nn.Sequential(nn.Conv2d(C_in, 64, 5), nn.ReLU(inplace=True), nn.Conv2d(64, 256, 3, stride=2), nn.ReLU(inplace=True), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(256, n_classes))


def net2(n_classes):
    end = net1(n_classes, 8)
    return nn.Sequential(nn.Conv2d(3, 8, 11), nn.BatchNorm2d(8), nn.ReLU(inplace=True), nn.Dropout2d(0.2), end)


def net3(n_classes):
    return nn.Sequential(nn.Conv2d(3, 32, 11), nn.ReLU(inplace=True), nn.AdaptiveMaxPool2d(5), nn.Flatten(), nn.Dropout(.3), nn.Linear(32*5**2, 128), nn.ReLU(inplace=True), nn.Linear(128, n_classes), nn.Softmax(dim=1))


print('Done')

# get score models

def get_scores(model):
    s1 = score_NTK(loader, model, device, 20)
    s2 = score_jacob(loader, model, device) * s1 * np.log(n_params(model))
    s3 = score_activations(loader, model, device)
    return float(s1), float(s2), float(-s3)

filename = 'scores_results.json'
if filename not in os.listdir('.'):
    # test models
    n_models = 5 + 3 + 3
    print_('getting scores:          0% ')
    n_classes = 10

    rand_scores = []
    loader = torch.utils.data.DataLoader(dataset, 8,  True) # custom batch size because some models use too much memory
    for i in range(5):
        for _ in range(repeat):
            clear_cache()                                   # clean GPU RAM
            model = get_random_net(n_classes).to(device)    # load model
            rand_scores.append(get_scores(model))           # score the model
        print_(f'{int(100*(i+1)/n_models)}% ')          # print run %


    popular_scores = []
    loader = torch.utils.data.DataLoader(dataset, 1,  True)
    for i, getter in enumerate([get_alexnet, get_resnet, get_vgg]):
        for _ in range(repeat):
            clear_cache()
            model = getter(n_classes).to(device)
            popular_scores.append(get_scores(model))
        print_(f'{int(100*(i+6)/n_models)}% ')


    hand_scores = []
    loader = torch.utils.data.DataLoader(dataset, 8,  True)
    for i, getter in enumerate([net1, net2, net3]):
        for _ in range(repeat):
            clear_cache()
            model = getter(n_classes).to(device)
            hand_scores.append(get_scores(model))
        print_(f'{int(100*(i+9)/n_models)}% ')

    # scores
    scores = [rand_scores, hand_scores, popular_scores]

    # save results
    with open(filename, 'w') as fout:
        json.dump(scores, fout)
else:
    with open(filename, 'r') as fin:
        scores = json.load(fin)

# get rid of eventual outliers
vals = scores[0]+scores[1]+scores[2]
minmax, perc = [], int(len(vals)/5)
for i in range(3):
    vals.sort(key=lambda x: x[i])
    diff = (vals[-perc][i]-vals[perc][i])/10
    minmax.append([vals[0][i]-diff, vals[-perc][i]+diff])

print('\n', scores)


# plot first 2 scores
colors = ['mediumspringgreen', 'deepskyblue', 'steelblue']
labels = ['random architecture', 'average architecture','popular architecture',]

for s, c, l in zip(scores, colors, labels):
    x = [v[0] for v in s]
    y = [v[1] for v in s]
    plt.scatter(x, y, color=c, label=l)

plt.title('scores for different architectures: better closer to origin')
plt.xlabel('Neural Tangent Kernel Score')
plt.ylabel('Difference Between Jacobians')
plt.xlim([minmax[0][0], minmax[0][1]/2])
plt.ylim([minmax[1][0], minmax[1][1]/2])
plt.legend()
plt.show()

# plot 3° feat
count = 0
for s, c, l in zip(scores, colors, labels):
    x = [count+i for i in range(len(s))]
    y = [v[2] for v in s]
    plt.scatter(x, y, color=c, label=l)
    count += len(s)

plt.title('relation between architectures and the third score')
plt.ylabel('Linear Region Score')
plt.ylim(minmax[2])
plt.legend()
plt.show()


fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)


for s, c, l in zip(scores, colors, labels):
    x = [(v[0]-minmax[0][0])/(minmax[0][1]-minmax[0][0]) for v in s]
    y = [(v[1]-minmax[1][0])/(minmax[1][1]-minmax[1][0]) for v in s]
    z = [(v[2]-minmax[2][0])/(minmax[2][1]-minmax[2][0]) for v in s]

    sizes = []
    for v in s:
        tot=5
        for v2 in vals:
            tot+= int(v[0]<v2[0])+int(v[1]<v2[1])+int(v[2]<v2[2])
        sizes.append(float(tot))

    ax.scatter(x,y,z, s=sizes, c=c, label=l)
ax.scatter([0],[0],[0], s=120, c='black')

ax.set_xlabel('NTK')
ax.set_ylabel('Jacob')
ax.set_zlabel('Regions')

plt.xlim([0, 1])
plt.ylim([0, 1])
ax.set_zlim3d([0, 1])
plt.legend()
plt.show()


"""   NN  popular scores

[(91.92739868164062, 27.72519, 45518.2031),
(102.24115753173828, 27.724752, 50624.3008),
(95.63170623779297, 27.72513, 47352.3047),
(102.0496597290039, 27.725193, 50530.2852),
(1653.3934326171875, 27.713036, 743657.625)),
(2025.8201904296875, 27.709034, 911034.875)),
(1393.7567138671875, 27.71905, 627015.125)),
(1646.406982421875, 27.713589, 740530.125)),
(52.73725128173828, 27.72586, 27365.6777),
(37.14685821533203, 27.725842, 19275.7148),
(29.311800003051758, 27.725855, 15210.0664),
(29.89641571044922, 27.725832, 15513.4150)]
"""