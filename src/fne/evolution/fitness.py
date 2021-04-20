import torch
from torch.utils import DataLoader

from .utils import get_conf
from .datasets import get_dataset
from ..genotopheno import LearnableCell


def fitness_score(genotype, C_in, search_space, original_dataset, max_distance, distance):
    arch, _, dataset = get_conf(genotype)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neural_net = LearnableCell(genotype, C_in, search_space).to(device)
    dataset = get_dataset(dataset, original_dataset, max_distance, distance)
    loader = DataLoader(dataset, batch_size=64)

    score1 = score_NTK(loader, neural_net, device)
    score2 = score_linear(loader, neural_net, device)
    score3 = score_params(neural_net)

    return (score1, score2, score3)


def score_NTK(dataloader: DataLoader, neural_net: torch.nn.Module, device, samples_to_use=200):
    neural_net.eval()

    grads = []
    for i, (inputs, targets) in enumerate(loader):
        # check how many samples have been processed
        if samples_to_use <= 0:
            break
        samples_to_use -= inputs.shape[0].item()

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
    return np.nan_to_num(
        (eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)


def score_linear(dataloader: DataLoader, neural_net: torch.nn.Module):
    # https://github.com/VITA-Group/TENAS/blob/main/lib/procedures/linear_region_counter.py
    pass


def score_params(neural_net: torch.nn.Module):
    pass
