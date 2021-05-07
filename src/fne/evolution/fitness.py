import torch
from torch.utils.data import DataLoader
import numpy as np

from .utils import get_conf
from .datasets import get_dataset
from ..genotopheno import LearnableCell

import gc


def fitness_score(genotype, C_in, search_space, original_dataset, max_distance, distance):
    gc.collect()
    torch.cuda.empty_cache()
    _, _, dataset = get_conf(genotype)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_net = LearnableCell(C_in, genotype, search_space).to(device)
    dataset = get_dataset(dataset, original_dataset, max_distance, distance)
    loader = DataLoader(dataset, batch_size=64)

    # sometimes it fails & i have no idea why
    try:
        score1 = score_NTK(loader, neural_net, device)
    except Exception as e:
        score1 = 1e10

    score2 = score_linear(loader, neural_net, device)
    # TODO: replace it with a new score in the future
    score3 = n_params(neural_net) * score1 * score2

    return (score1, score2, score3)


def score_NTK(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=200):
    """fitted nets minimize this score (the lowest the best)"""
    neural_net.eval()

    grads = []
    for i, (inputs, targets) in enumerate(dataloader):
        # check how many samples have been processed
        if samples_to_use <= 0:
            break
        samples_to_use -= inputs.shape[0]

        # pass input through network
        inputs = inputs.to(device)
        neural_net.zero_grad()
        inputs_ = inputs.clone().to(device)
        logit = neural_net(inputs_)

        # for each node in the output layer
        for _idx in range(len(inputs_)):
            # calculate gradient on the weights
            logit[_idx:_idx +
                  1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in neural_net.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach())
            # append the gradient vector for that node/sample
            grads.append(torch.cat(grad, -1))
            neural_net.zero_grad()
            torch.cuda.empty_cache()
    # make matrix of gradients
    grads = torch.stack(grads, 0)
    # ntk_ij = sum_c[ grads_ic * grads_jc ] = grads @ grads.transpose(0,1)
    # ntk_ij is symmetric, ntk_ij is similar to cosine_similarity_between_gradi_gradj, but without normalization
    ntk = torch.einsum('nc,mc->nm', [grads, grads])
    # get eigenvalues
    eigenvalues, _ = torch.symeig(ntk)
    # if big difference between biggest & smallest --> GOOD
    # I think the intuition is like: "there is regularity in how the gradients are similar between each other"
    return np.nan_to_num(
        (eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)


# https://github.com/BayesWatch/nas-without-training/blob/8ba0313ea1b6038e6d0c6822031a100135715e2a/score_networks.py
# the code is a pretty bad (there were 2 score functions, but i only understood this one (and changed it))
def score_linear(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=200):
    """fitted nets minimize this score (the lowest the best)"""
    neural_net.eval()

    jacobs, inps = [], []
    for i, (inputs, targets) in enumerate(dataloader):
        # check how many samples have been processed
        if samples_to_use <= 0:
            break
        samples_to_use -= inputs.shape[0]

        inps += [inp for inp in inputs]
        grads = []
        neural_net.zero_grad()
        inputs = inputs.to(device)
        inputs.requires_grad_(True)

        # get the gradient of the inputs wrt. the output of the i° neuron in the output layer
        outputs = neural_net(inputs)
        for i in range(outputs.size(1)):
            (outputs[:, i]).sum().backward(retain_graph=True)
            # (Batch, ch, H, W).view(inputs.size(0), -1)
            grads.append(inputs.grad.detach().view(inputs.size(0), -1))
            inputs.grad.zero_()
        grads = torch.cat([g.unsqueeze(0).to('cpu') for g in grads], dim=0)
        ## grads.shape = (output_neurons, batch_size, inputs)

        jacobs.append(grads)  # then concat on batch size dimension

    # jacobs_i,j,k = gradient for sample i,  of the function "output neuron j" for the "input pixel k"               (not really a pixel because channels are separated...)
    jacobs = torch.cat(jacobs, dim=1).transpose(0, 1)

    nsamples = jacobs.shape[0]
    # get correlations between the jacobians
    K = torch.zeros((nsamples, nsamples))
    for i in range(nsamples):
        for j in range(0, i+1):
            x = jacobs[i, :, :]
            y = jacobs[j, :, :]
            corr = (x*y).mean() / (x.mean()*y.mean())
            # --> improvable measure
            input_simil = 1. - ((inps[i]-inps[j])**2).mean()
            K[i, j] = corr * input_simil
            K[j, i] = corr * input_simil

    # determinant to summarize
    det = torch.det(K)
    # if we have many linear regions
    # --> the rows of K should be different between each other
    # --> high variance --> high determinant

    # return a minus score (so we have to minimize it)
    return np.nan_to_num((-torch.log(det)).detach().numpy(), copy=True, nan=100000.0)


def n_params(neural_net: torch.nn.Module):
    """number of parameters: the lowest the best"""
    return sum(p.numel() for p in neural_net.parameters())
