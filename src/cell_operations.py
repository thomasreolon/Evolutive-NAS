import torch
import torch.nn as nn

OPS = {
    'dil_conv_3x3': lambda C_in, C_out, stride, affine: SepConv((3,3), C_in, C_out, stride, affine),
    'dil_conv_5x5': lambda C_in, C_out, stride, affine: SepConv((5,5), C_in, C_out, stride, affine),
    'dil_conv_7x7': lambda C_in, C_out, stride, affine: SepConv((7,7), C_in, C_out, stride, affine),
    'skip_connect': lambda C_in, C_out, stride, affine: SkipConn(      C_in, C_out, stride, affine),
    'clinc_3x3':    lambda C_in, C_out, stride, affine: ConvLinConv(True, C_in, C_out, stride, affine),
    'clinc_7x7':    lambda C_in, C_out, stride, affine: ConvLinConv(False, C_in, C_out, stride, affine),
    'avg_pool_3x3': lambda C_in, C_out, stride, affine: Pool('avg' , C_in, C_out, stride, affine),
    'max_pool_3x3': lambda C_in, C_out, stride, affine: Pool('max' , C_in, C_out, stride, affine),
}



class SepConv(nn.Module):
    def __init__(self, kernel_size, C_in, C_out, stride, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=1, dilation=1, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x):
        return self.op(x)


class Linearize(nn.Module):
    def __init__(self, kernel_size, mode, C_in, C_out, stride, affine):
        super().__init__()
        op = nn.AdaptiveAvgPool2d(1) if mode == 'avg' else nn.AdaptiveMaxPool2d(1)
        self.layers = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            op,
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)


class ConvLinConv(nn.Module):
    def __init__(self, is_3x3, C_in, C_out, stride, affine):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        if is_3x3:
            lin_size, kernel = 128, (3,3)
            self.tolinear = Linearize(kernel, 'avg', C_in, lin_size, 1, affine)
        else:
            lin_size, kernel = 512, (7,7)
            self.tolinear = Linearize(kernel, 'avg', C_in, lin_size, 5, affine)
        self.layer = nn.Linear(lin_size, C_out)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel, stride=stride)

    def forward(self, x):
        l = self.tolinear(x)
        l = self.layer(l)
        l = l.view(l.shape[0], -1, 1, 1)
        r = self.conv(x)
        return l*r

class Pool(nn.Module):
    def __init__(self, mode, C_in, C_out, stride, affine):
        super().__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_out, (3,3), stride=stride, padding=0, dilation=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )
        if   mode == 'avg': self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max': self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else              : raise ValueError('Invalid mode={:} in POOLING'.format(mode))

    def forward(self, x):
        if self.preprocess: x = self.preprocess(x)
        return self.op(x)



class SkipConn(nn.Module):
    def __init__(self, C_in, C_out, stride, affine):
        super().__init__()
        if C_in==C_out:
            self.changech = lambda x: x 
        else: 
            self.changech = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        out = self.changech(x)
        out = self.bn(out)
        return out


if __name__=='__main__':
    confs = {'C_in':3, 'C_out':8, 'stride':1, 'affine':True}

    for name, layer in OPS.items():
        print('\ntrying layer', name)
        net = layer(**confs)
        x = torch.rand((16,confs['C_in'],32,32))
        y = net(x)
        print('output: ', y.shape,  'hash', hash(net))

