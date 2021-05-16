
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from fne.genotopheno.cell_operations import OPS
from fne.genotopheno import LearnableCell, DARTSNetwork
from fne.evolution import Mutations, get_conf

assert torch.cuda.is_available()

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                    'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
n_epochs = 10

def main():
    # dataloaders
    trainl, testl = get_loaders()

    # neural net 1
    geno1 = get_random_genotype()
    geno2 = get_random_genotype()
    neural_net1 = DARTSNetwork(3, len(classes), [geno1,geno2,geno1,geno2], search_space, depth=2)
    neural_net1.to('cuda')

    for e in range(n_epochs):
        print(f'------------------- epoch {e} ---------------')
        tr_loss = train_nn(neural_net1, trainl)
        te_loss = train_nn(neural_net1, testl, train=False)
        print(f'train loss = {tr_loss},       test loss = {te_loss}')
    test_nn(neural_net1, testl)

    # neural net 2
    neural_net2 = Net(len(classes))
    neural_net2.to('cuda')

    for e in range(n_epochs):
        print(f'------------------- epoch {e} ---------------')
        tr_loss = train_nn(neural_net2, trainl)
        te_loss = train_nn(neural_net2, testl, train=False)
        print(f'train loss = {tr_loss},       test loss = {te_loss}')
    test_nn(neural_net2, testl)


def train_nn(neural_net, loader, train=True):
    tot_loss, cc = 0, 0
    # optimizer & loss
    loss_fn = torch.nn.CrossEntropyLoss()

    if train:
        optimizer = torch.optim.Adam(neural_net.parameters())
        neural_net.train()
        torch.set_grad_enabled(True)
    else: 
        neural_net.eval()
        torch.set_grad_enabled(False)


    for inputs, targets in loader:
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')

        outputs = neural_net(inputs)
        loss = loss_fn(outputs, targets)
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        tot_loss+=loss.item()
        cc+=1
    
    return tot_loss/cc

def test_nn(neural_net, loader):
    conf_mat = torch.zeros((len(classes), len(classes)), dtype=torch.float)
    neural_net.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to('cuda')

            outputs = neural_net(inputs).to('cpu')
            ipos = targets.to('cpu')
            _, jpos = outputs.max(dim=1)
            for i,j in zip(ipos,jpos):
                conf_mat[i,j] += 1.
    n_params = sum(p.numel() for p in neural_net.parameters() if p.requires_grad)
    acc = torch.diag(conf_mat).sum() / conf_mat.sum()
    avg = (torch.diag(conf_mat) / conf_mat.sum(dim=1))
    worst = avg.min()
    avg = avg.mean()

    print(f"============================= neural net with {n_params} params ==============================")
    print(f"accuracy={acc.item()},      avg.accuracy={avg.item()},      worstclassaccuracy={worst.item()}")



def get_loaders():
    augtransform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=augtransform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=2)
    return trainloader, testloader

class Net(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn = nn.BatchNorm2d(6, affine=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16, affine=True)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(120, 84)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        # [conv->norm->activ->pool] *2
        x = self.bn(self.conv1(x))
        x = self.pool(torch.relu(x))
        x = self.bn2(self.conv2(x))
        x = self.pool(torch.relu(x))
        # [linear->activ->drop] *2
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.relu(self.fc2(x))
        x = self.drop2(x)
        # linear
        x = self.fc3(x)
        return x



def get_random_genotype():
    genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
    mutator = Mutations(search_space)
    for i in range(5):
        genotype = mutator(genotype)
    return genotype

if __name__=='__main__':
    main()