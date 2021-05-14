import torch
from torch.utils.data import DataLoader
import numpy as np

from .utils import get_conf, clear_cache
from .datasets import get_dataset
from ..genotopheno import LearnableCell

import gc


def fitness_score(genotype, C_in, search_space, original_dataset, max_distance, distance):
    _, _, dataset = get_conf(genotype)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_net = LearnableCell(C_in, genotype, search_space).to(device)
    dataset = get_dataset(dataset, original_dataset, max_distance, distance)
    # TODO: add config gpu RAM
    loader = DataLoader(dataset, batch_size=64)

    # sometimes it fails & i have no idea why
    try:
        score1 = score_NTK(loader, neural_net, device)
    except Exception as e:
        score1 = 1e10

    score2 = score_linear(loader, neural_net, device)
    # TODO: replace it with a new score in the future
    score3 = (torch.log(n_params(neural_net)) * score1 * score2).item()

    return (score1, score2, score3)


def score_NTK(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=20, max_ram=8):
    """fitted nets minimize this score (the lowest the best)"""
    neural_net.eval()
    remaining = samples_to_use
    grads = []
    for i, (inputs, targets) in enumerate(dataloader):
        clear_cache()
        # check how many samples have been processed
        if remaining <= 0:
            break
        remaining -= inputs.shape[0]

        # pass input through network
        inputs = inputs.to(device)
        neural_net.zero_grad()
        inputs_ = inputs.clone().to(device)
        logit = neural_net(inputs_)

        # for each node in the output layer
        for _idx in range(len(inputs_)):
            clear_cache()
            # calculate gradient on the weights
            retain = _idx!=len(inputs_)-1
            logit[_idx:_idx +
                  1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=retain)
            grad = []
            params_count, max_ = 0, max_ram*125000000/int(samples_to_use/len(inputs_)+1)/len(inputs_) ######## i have 8GB gpu ram --> limit max number of params
            for name, W in neural_net.named_parameters():
                if 'weight' in name and W.grad is not None:
                    if params_count + W.numel() > max_: break
                    grad.append(W.grad.view(-1).detach())
                    params_count += W.numel()
            # append the gradient vector for that node/sample
            grads.append(torch.cat(grad, -1))
            neural_net.zero_grad()
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
        (eigenvalues[-1] / eigenvalues[0]).abs().item(), copy=True, nan=100000.0)


# https://github.com/BayesWatch/nas-without-training/blob/8ba0313ea1b6038e6d0c6822031a100135715e2a/score_networks.py
# the code is a pretty bad (there were 2 score functions, but i only understood this one (and changed it))
def score_linear(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=40):
    """fitted nets minimize this score (the lowest the best)"""
    neural_net.eval()

    jacobs, inps = [], []
    for i, (inputs, targets) in enumerate(dataloader):
        clear_cache()
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
            grads.append(inputs.grad.detach().view(inputs.size(0), -1).clone())
            inputs.grad.zero_()
        grads = torch.stack(grads)
        ## grads.shape = (output_neurons, batch_size, inputs)

        jacobs.append(grads)  # then concat on batch size dimension

    # jacobs_i,j,k = gradient for sample i,  of the function "output neuron j" for the "input pixel k"               (not really a pixel because channels are separated...)
    jacobs = torch.cat(jacobs, dim=1).transpose(0, 1)

    nsamples = jacobs.shape[0]
    # get correlations between the jacobians
    K = torch.zeros((nsamples, nsamples))
    for i in range(nsamples):
        for j in range(0, i+1):
            x = jacobs[i, :, :] - jacobs[i, :, :].mean()
            y = jacobs[j, :, :] - jacobs[j, :, :].mean()
            corr = (x*y).sum() / ((x**2).sum()*(y**2).sum())**(1/2)
            # --> improvable measure
            input_simil = 4. - ((inps[i]-inps[j])**2).mean()
            K[i, j] = corr * input_simil
            K[j, i] = corr * input_simil

    # determinant to summarize
    det = torch.det(K).abs()
    score = np.nan_to_num(torch.log(det).detach().numpy(), copy=True, nan=100000.0)
    if score < 0: score = 100  # (workaround) for some reasons, networks with many skip-connect get negative scores
    return score


def n_params(neural_net: torch.nn.Module):
    """number of parameters: the lowest the best"""
    tot = sum(p.numel() for p in neural_net.parameters())
    return torch.tensor([tot], dtype=torch.float)
