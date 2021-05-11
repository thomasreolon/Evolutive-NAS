import torch
import torch.nn as nn

from .cells import LearnableCell


class Flatten(nn.Module):
    """in my pytorch version it is not implemented =/"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class VisionNetwork(nn.Module):
    def __init__(self, C_in, n_classes, population, search_space, depth=3):
        super().__init__()

        assert len(population) % depth == 0
        width = int(len(population) / depth)

        self.alphas = nn.Parameter(torch.ones((depth, width)))
        self.layers = nn.ModuleList()
        self.depth = depth
        self.population = population

        C_ins = [C_in] + [0 for _ in range(depth)]
        for i, genotype in enumerate(population):
            prev_l = int(i/width)
            cell = LearnableCell(C_ins[prev_l], genotype, search_space)
            self.layers.append(cell)
            C_ins[1+prev_l] += cell.C_out

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(C_ins[-1]),
            nn.Dropout2d(),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(C_ins[-1], n_classes),
        )

    def forward(self, x):
        inputs, width = [x], int(len(self.layers)/self.depth)
        weights = -torch.log_softmax(self.alphas, dim=1)
        device = next(self.parameters()).device

        for i in range(self.depth):
            tmp = []
            for j in range(width):
                cell = self.layers[i*width+j]
                if weights[i, j] > 0.01:
                    res = cell(inputs[i]) * weights[i, j]
                    tmp.append(res)
                else:
                    s = inputs[i].shape
                    h,w = (int((s[2]-1)/2), int((s[3]-1)/2)) if cell.do_pool else (int(s[2]), int(s[3]))
                    tmp.append(torch.zeros(
                        (s[0], cell.C_out, h, w), device=device))

            tmp = torch.cat(tmp, dim=1)
            noise = torch.rand(tmp.shape, device=device)
            inputs.append(tmp + noise*2e-5)

        return self.classifier(inputs[-1])
