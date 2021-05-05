from torch.utils.data import Dataset
import torch
import random
import math

datasets_cache = {}



class VisionDataset(Dataset):
    """
    we need less samples to test this dataset
    """
    def __init__(self, number, original_dataset, max_distance, distance=None):
        super().__init__()
        self.n = number
        self.data = original_dataset
        self.ids = []
        self.max_distance = max_distance
        self.distance = distance or (lambda x,y: (torch.abs(torch.tensor(x)-torch.tensor(y))).sum())

        # take a subset of the original_dataset
        self.select_ids()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idx = self.ids[idx]
        return self.data[idx]


    def select_ids(self):
        """
        if 'number' is low  --> dataset will be random samples from original
        if 'number' is high --> dataset will be unbalanced over a class
        """
        i, order = 0, list(range(len(self.data)))
        random.shuffle(order)
        base_class = self.get_base_class()
        while len(self.ids)<200 and i<len(self.data):
            _, y = self.data[order[i]]
            dis = float(self.distance(base_class, y))
            prob = 0.4 - (dis/self.max_distance)*self.n/10
            if random.random() < prob:
                self.ids.append(order[i])
            i = (i+1)%len(self.data)

    def get_base_class(self):
        """
        points will be choosen near this class:
        40% is a random sampled class
        60% is a weak class
        """
        dist = self.distance
        l = len(self.data)
        if random.random()>0.6:
            # random class
            _, base_class = self.data[random.randint(0,l)]
        else:
            # probably weak class
            _, bc1 = self.data[random.randint(0,l)]
            _, bc2 = self.data[random.randint(0,l)]
            _, bc3 = self.data[random.randint(0,l)]
            best_score, base_class = -1, None
            n_tests = 10+int(math.log(self.max_distance))
            for i in range(n_tests):
                _, bc = self.data[random.randint(0,l)]
                score = dist(bc, bc1) + dist(bc, bc2) + dist(bc, bc3)
                if score > best_score:
                    best_score, base_class = score, bc
        return base_class




def get_dataset(number, original_dataset, max_distance, distance=None):
    if number in datasets_cache:
        dataset = datasets_cache[number]
    else:
        dataset = VisionDataset(number, original_dataset, max_distance, distance)
        datasets_cache[number] = dataset
    return dataset
    


