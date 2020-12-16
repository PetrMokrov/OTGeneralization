import torch
import torch.nn as nn

class QuadraticTransportCost:

    def __init__(self):
        pass

    def __call__(self, x, y):
        b_size = x.size(0)
        return torch.sqrt(((x - y)**2).view(b_size, -1).sum(-1))