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
    loader = DataLoader(dataset, batch_size=64)

    # sometimes it fails & i have no idea why
    try:
        score1 = score_NTK(loader, neural_net, device)
    except Exception as e:
        score1 = 1e10

    score2 = score_jacob(loader, neural_net, device)

    score3 = score_activations(loader, neural_net, device)

    return (score1, score2, score3)


### from https://github.com/VITA-Group/TENAS/blob/main/lib/procedures/ntk.py
# some changes not to run out of memory
def score_NTK(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=20):
    """fitted nets minimize this score (the lowest the best)"""

    # calculate how much ram
    free_mem  = torch.cuda.get_device_properties(0).total_memory -torch.cuda.memory_allocated(0)
    max_ram = int(free_mem/1e9)  ## how many GB are free (hoping that no other application is using the RAM)

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
def score_jacob(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=40):
    """fitted nets minimize this score (the lowest the best)"""
    neural_net.eval()

    jacobs, inps = [], []
    for i, (inputs, _) in enumerate(dataloader):
        clear_cache()
        # check how many samples have been processed
        if samples_to_use <= 0:
            break
        samples_to_use -= inputs.shape[0]

        inps += [inp for inp in inputs]
        grads = []
        neural_net.zero_grad()
        inputs = inputs.to(device).requires_grad_(True)

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
            corr = 0
            for k in range(jacobs.size(1)):
                x = jacobs[i, k, :] - jacobs[i, k, :].mean()
                y = jacobs[j, k, :] - jacobs[j, k, :].mean()
                corr += (x*y).sum() / ((x**2).sum()*(y**2).sum())**(1/2)

            # TODO: better way to measure similarity between 2 inputs
            input_simil = 4. - ((inps[i]-inps[j])**2).mean()
            K[i, j] = corr * input_simil
            K[j, i] = corr

    # determinant to summarize
    _, ld = np.linalg.slogdet(K.cpu().numpy())
    score = np.nan_to_num(ld, copy=True, nan=100000.0)
    return score


def n_params(neural_net: torch.nn.Module):
    """number of parameters: the lowest the best"""
    return sum(p.numel() for p in neural_net.parameters())

# https://github.com/BayesWatch/nas-without-training/blob/5368686cb0b740d5a779bb787d6fa2d5fe5cbe1f/score_networks.py
#### Not sure about this code...
def score_activations(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=256):
    """fitted nets minimize this score (the lowest the best)"""
    neural_net.eval()
    def counting_forward_hook(module, inp, out):
        try:
            if isinstance(inp, tuple): inp = inp[0]
            # flatten input
            inp = inp.view(inp.size(0), -1)
            # where there are activations
            x = (inp > 0).float()

            # correlation between samples in the same batch
            K = x @ x.t()
            K2 = (1.-x) @ (1.-x.t())
            neural_net.K +=  K.cpu().numpy() + K2.cpu().numpy()
        except:
            pass
   
    
    for _, module in neural_net.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_forward_hook(counting_forward_hook)

    neural_net.K = None
    neural_net, s = neural_net.to(device), []
    with torch.no_grad():
        for inputs, _ in dataloader:
            clear_cache()
            if samples_to_use <= 0:
                break
            samples_to_use -= inputs.shape[0]

            # sum of similarities
            neural_net.K = np.zeros((inputs.shape[0], inputs.shape[0]))

            # during forward we calculate K
            neural_net(inputs.to(device))

            # abs(log determinant is the score)
            _, ld = np.linalg.slogdet(neural_net.K)
            s.append(ld)
        # average over some batches
        score = np.mean(s)
    ## we want to minimize, so negative
    return score


