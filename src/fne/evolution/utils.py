def get_conf(genotype):
    """transform a genotype(string) in some variables"""
    architecture, evol_strattegy = genotype.split('--')
    architecture = [[int(x) for x in conn.split('|')]
                    for conn in architecture.split('  ')]

    use_shared, dataset = evol_strattegy.split('  ')
    use_shared, dataset = int(use_shared), int(dataset)
    return architecture, use_shared, dataset



def encode_conf(architecture, use_shared, dataset):
    architecture = ['|'.join([str(x) for x in a]) for a in architecture]
    architecture = '  '.join(architecture)
    return f'{architecture}--{use_shared}  {dataset}'