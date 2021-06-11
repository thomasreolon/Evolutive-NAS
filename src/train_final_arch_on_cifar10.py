import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from fne.genotopheno import EvaluationNetwork
from fne.evolution.utils import clear_cache

import os, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))

assert torch.cuda.is_available()




# dataset
transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,4,padding_mode='reflect') ,transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

name='results_nn_ga.pth'
# NN setup
if name=='results_nn_ga.pth':
    with open('cifar_results.txt', 'r') as fin:
        population = json.load(fin)['populations'][-1]
        population.sort(key=lambda x: len(x))
        population = population[:2]
    print(population)
    
    search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
    neural_net   = EvaluationNetwork(3, 10, population, search_space)
    print(sum(p.numel() for p in neural_net.parameters()))
    if name in os.listdir('.'):
        print('loading_checkpoint')
        neural_net.load_state_dict(torch.load(name))
else:
    name = 'results_nn_res.pth'
    neural_net = torchvision.models.resnet18()
    neural_net.fc = nn.Linear(512, 10)
    if name in os.listdir('.'):
        neural_net.load_state_dict(torch.load(name))
neural_net = neural_net.to('cuda')

# loader, loss, ...
train_loader = DataLoader(trainset, 64, True)
optimizer    = torch.optim.Adam(neural_net.parameters(), lr=.02, weight_decay=1e-5)
loss_fn      = nn.CrossEntropyLoss()

losses = [1.]
neural_net.train()
for e in range(7):
    tot_loss, cc = 0, 0
    for inps, targs in train_loader:
        inps, targs = inps.cuda(), targs.cuda()

        outs = neural_net(inps)
        loss = loss_fn(outs, targs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tot_loss += loss.item()
        cc += inps.shape[0]
    losses.append(tot_loss/cc)
    print(f'loss epoch {e}: {losses[-1]:.3e}')

    torch.save(neural_net.state_dict(), 'trained-'+name)
    clear_cache()

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*(0.99 + (losses[-2]-losses[-1])/losses[-2])


# test set 
print('testing...')
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, 16, True)

for k, loader in enumerate((test_loader, train_loader)):
    CM = torch.zeros((10,10))
    neural_net.eval()
    for inps, targs in loader:
            inps = inps.cuda()
            outs = neural_net(inps)
            _, outs = outs.max(dim=1)
            for i,j in zip(targs, outs):
                CM[i,j] += 1.

    recalls = torch.diag(CM)/CM.sum(dim=1)
    accuracy = torch.trace(CM)/CM.sum()

    print(k and "trainset" or "testset")
    print('\nRECALLS', recalls.tolist(), '\nACCURACY', accuracy.item(), '\n')


