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

        self.alphas = nn.Parameter(torch.ones((depth, wide)))
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
        )

    def forward(self, x):
        inputs, wide = [x], int(len(self.layers)/self.depth)
        weights = nn.functional.softmax(self.alphas, dim=1)

        for i in range(self.depth):
            tmp = []
            for j in range(wide):
                cell = self.layers[i*self.depth+j]
                if weights[i,j]>0.01:
                    res = cell(inputs[i]) * weights[i,j]
                    tmp.append(res)
                else:
                    s = inputs[i].shape
                    tmp.append(torch.zeros((s[0], cell.C_out, int((s[2]-1)/2), int((s[3]-1)/2))))

            tmp = torch.cat(tmp, dim=1)
            noise = torch.rand(tmp.shape, device=next(self.parameters()).device)
            inputs.append(tmp + noise*2e-5)

        return self.classifier(inputs[-1])
