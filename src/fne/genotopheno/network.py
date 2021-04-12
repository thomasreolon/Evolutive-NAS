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
        wide = int(len(population) / depth)

        self.alphas = torch.ones((depth, wide), requires_grad=True)
        self.layers = nn.ModuleList()
        self.depth = depth

        C_ins = [C_in] + [0 for _ in range(depth)]
        for i, genotype in enumerate(population):
            prev_l = int(i/wide)
            cell = LearnableCell(C_ins[prev_l], genotype, search_space)
            self.layers.append(cell)
            C_ins[1+prev_l] += cell.C_out

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(C_ins[-1]),
            nn.Dropout2d(),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(C_ins[-1], n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        inputs, wide = [x], int(len(self.layers)/self.depth)

        for i in range(self.depth):
            tmp = []
            for j in range(wide):
                cell = self.layers[i*self.depth+j]
                res = cell(inputs[i])
                tmp.append(res)

            tmp = torch.cat(tmp, dim=1)
            noise = torch.rand(tmp.shape)
            inputs.append(tmp + noise*1e-7)

        return self.classifier(inputs[-1])
