import torch
import torch.nn as nn

class FullyConnectedMLP(nn.Module):

    def __init__(
        self, input_dim, hiddens, output_dim, 
        apply_batch_norm=True, device='cuda'):

        assert isinstance(hiddens, list)
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.apply_batch_norm = apply_batch_norm
        self.device = device
        self.hiddens = hiddens

        model = []
        inputs = [self.input_dim] + self.hiddens
        outputs = self.hiddens + [self.output_dim]

        for inp, outp in zip(inputs, outputs):
            model.append(nn.Linear(inp, outp))
            if self.apply_batch_norm:
                model.append(nn.BatchNorm1d(outp))
            model.append(nn.ReLU())

        del model[-1] # delete last ReLU
        self.net = nn.Sequential(*model)
        self.to(self.device)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x).view(batch_size, self.output_dim)

class ToyGenerator(FullyConnectedMLP):

    def __init__(self, z_dim, data_dim, hiddens, device='cuda'):
        super().__init__(
            z_dim, hiddens, data_dim, 
            apply_batch_norm=True, device=device)
        
        self.noise = torch.distributions.Uniform(0., 1.)
        
    def sample(self, n_samples):
        z = self.noise.sample([n_samples, self.input_dim]).to(self.device)
        return self.forward(z)

class ToyCritic(FullyConnectedMLP):

    def __init__(self, data_dim, hiddens, device='cuda'):

        super().__init__(
            data_dim, hiddens, 1, 
            apply_batch_norm=True, device=device)
    
    def forward(self, x):
        return torch.tanh(super().forward(x))


class ToyMover(FullyConnectedMLP):

    def __init__(self, data_dim, hiddens, device='cuda'):

        super().__init__(
            data_dim, hiddens, data_dim, 
            apply_batch_norm=False, device=device)

        
