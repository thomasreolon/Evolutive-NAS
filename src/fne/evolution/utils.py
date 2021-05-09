import random, gc, torch, sys
rnd = random.randint

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


def correct_genotype(genotype):
    architecture, use_shared, dataset = get_conf(genotype)
    if dataset<0 or dataset>10: dataset = rnd(0,10) 
    if use_shared not in (0,1): use_shared = rnd(0,1) 

    tot, prev = 0, -1
    for i, block in enumerate(architecture):
        tot += sum(block)
        act = int(((i+1)*2)**0.5)
        if act*(act+1)//2 == i+1:
            if tot==0 and i+1==len(architecture) and i>0:
                # remove last layer if it is empty
                less_depth   = int(act*(act-1)//2)
                architecture = architecture[:less_depth]
            elif tot==0:
                # add a connection randomly
                c, s     = rnd(0,len(block)-1), rnd(1, 7)
                block[c] = s

            tot = 0
    return encode_conf(architecture, use_shared, dataset)

# general utils

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def print_memory():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass

def print_(*args, **kw):
    print(*args, **kw, end='')
    sys.stdout.flush()