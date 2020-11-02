import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from copy import copy

from .layers import ConvexQuadratic

class LinDenseICNN(nn.Module):
    '''
    Fully Connected ICNN which follows the [Makkuva et.al.] article:
    (https://arxiv.org/pdf/1908.10962.pdf)
    '''

    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[64, 64, 64, 64], 
        activation=nn.LeakyReLU(0.2), device='cuda'):

        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.in_dim = in_dim
        self.device = device

        _hidden = copy(self.hidden_layer_sizes)
        w_sizes = zip(_hidden[:-1], _hidden[1:])

        self.W_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in w_sizes
        ])

        self.A_layers = nn.ModuleList([
            nn.Linear(self.in_dim, out_dim) 
            for out_dim in _hidden
        ])
        self.final_layer = nn.Linear(self.hidden_layer_sizes[-1], 1, bias=False)
        self.to(self.device)
    
    def forward(self, input):
        z = self.activation(self.A_layers[0](input))
        for a_layer, w_layer in zip(self.A_layers[1:], self.W_layers[:]):
            z = self.activation(a_layer(input) + w_layer(z))
        
        return self.final_layer(z)
    
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).to(self.device).float()
        )[0]
        return output      

    def convexify(self):
        for layer in self.W_layers:
            assert isinstance(layer, nn.Linear)
            layer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
    
    def relaxed_convexity_regularization(self):
        regularizer = 0.
        for layer in self.W_layers:
            assert isinstance(layer, nn.Linear)
            regularizer += layer.weight.clamp(max=0.).pow(2).sum()
        regularizer += self.final_layer.weight.clamp(max=0.).pow(2).sum()
        return regularizer
    
class DenseICNN(nn.Module):
    '''Fully Conncted ICNN with input-quadratic skip connections'''
    def __init__(
        self, in_dim, 
        hidden_layer_sizes=[32, 32, 32],
        rank=1, activation='celu', dropout=0.03,
        strong_convexity=1e-6, device='cuda'
    ):
        super(DenseICNN, self).__init__()
        self.device = device
        self.strong_convexity = strong_convexity
        self.hidden_layer_sizes = hidden_layer_sizes
        self.droput = dropout
        self.activation = activation
        self.rank = rank
        
        self.quadratic_layers = nn.ModuleList([
            nn.Sequential(
                ConvexQuadratic(in_dim, out_features, rank=rank, bias=True),
                nn.Dropout(dropout)
            )
            for out_features in hidden_layer_sizes
        ])
        
        sizes = zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:])
        self.convex_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features, bias=False),
                nn.Dropout(dropout)
            )
            for (in_features, out_features) in sizes
        ])
        
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], 1, bias=False)

    def forward(self, input):
        output = self.quadratic_layers[0](input)
        for quadratic_layer, convex_layer in zip(self.quadratic_layers[1:], self.convex_layers):
            output = convex_layer(output) + quadratic_layer(input)
            if self.activation == 'celu':
                output = torch.celu(output)
            elif self.activation == 'softplus':
                output = F.softplus(output)
            else:
                raise Exception('Activation is not specified or unknown.')
        
        return self.final_layer(output) + .5 * self.strong_convexity * (input ** 2).sum(dim=1).reshape(-1, 1)
    
    def push(self, input):
        output = autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((input.size()[0], 1)).to(self.device).float()
        )[0]
        return output    
    
    def convexify(self):
        for layer in self.convex_layers:
            for sublayer in layer:
                if (isinstance(sublayer, nn.Linear)):
                    sublayer.weight.data.clamp_(0)
        self.final_layer.weight.data.clamp_(0)
              
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape)

class ConvICNN128(nn.Module):
    def __init__(self, channels=3, device = 'cuda'):
        super(ConvICNN128, self).__init__()
        self.device = device

        self.first_linear = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
        )
        
        self.first_squared = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
        )
        
        self.convex = nn.Sequential(
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1),  
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            nn.Conv2d(128, 128, kernel_size=3,stride=2, bias=True, padding=1), 
            nn.CELU(),
            View(32* 8 * 8),
            nn.CELU(), 
            nn.Linear(32 * 8 * 8, 128),
            nn.CELU(), 
            nn.Linear(128, 64),
            nn.CELU(), 
            nn.Linear(64, 32),
            nn.CELU(), 
            nn.Linear(32, 1),
            View()
        ).to(device=self.device)

    def forward(self, input): 
        output = self.first_linear(input) + self.first_squared(input) ** 2
        output = self.convex(output)
        return output
    
    def push(self, input):
        return autograd.grad(
            outputs=self.forward(input), inputs=input,
            create_graph=True, retain_graph=True,
            only_inputs=True, grad_outputs=torch.ones(input.size()[0]).to(device=self.device).float()
        )[0]
    
    def convexify(self):
        for layer in self.convex:
            if (isinstance(layer, nn.Linear)) or (isinstance(layer, nn.Conv2d)):
                layer.weight.data.clamp_(0)