
if __name__ == '__main__':
    confs = {'C_in': 3, 'C_out': 8, 'stride': 1, 'affine': True}

    for name, layer in OPS.items():
        print('\ntrying layer', name)
        net = layer(**confs)
        x = torch.rand((16, confs['C_in'], 32, 32))
        y = net(x)
        print('output: ', y.shape,  'hash', hash(net))

# check if everything works fine
if __name__ == '__main__':
    genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
    search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                    'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}

    net = LearnableCell(3, genotype, search_space)

    x = torch.rand((16, 3, 32, 32))
    y = net(x)
    print(y.shape)


# check if it works
if __name__ == '__main__':
    genotype = '0|0|2|0|0|2|0|0  1|0|0|1|1|0|0|0  0|1|0|0|0|0|2|1--1  7'
    search_space = {'dil_conv_3x3', 'dil_conv_5x5', 'dil_conv_7x7',
                    'skip_connect', 'clinc_3x3', 'clinc_7x7', 'avg_pool_3x3',  'max_pool_3x3'}
    population = [genotype]*4

    net = VisionNetwork(3, 5, population, search_space, 2)
    x = torch.rand(16, 3, 128, 64)
    y = net(x)
    print(y.shape)
    print(y[0])
