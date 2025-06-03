import math
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions
from torch.nn.parameter import Parameter

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class InvertiblePLU(nn.Module):
    def __init__(
        self,
        features: int,
    ):
        super().__init__()
        self.features = features
        w_shape = (self.features, self.features)
        w = torch.empty(w_shape)
        nn.init.orthogonal_(w)
        P, L, U = torch.linalg.lu(w)

        self.s = nn.Parameter( torch.diag(U) )
        self.U = nn.Parameter( U - torch.diag(self.s) )
        self.L = nn.Parameter( L )
        
        self.P = nn.Parameter(P, requires_grad=False)
        self.P_inv = nn.Parameter(torch.linalg.inv(P), requires_grad=False)

    def forward(self, x):
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.features, device=x.device)
        U = torch.triu(self.U, diagonal=1)
        s = self.s
        
        W = self.P @ L @ (U + torch.diag(s))
        
        z = x @ W
        logdet = torch.sum( torch.log( torch.abs(s) ), dim=0, keepdim=True )
        return z, logdet

    def reverse(self, x):
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.features, device=x.device)
        U = torch.triu(self.U, diagonal=1)
        s = self.s

        eye = torch.eye(self.features, device=x.device, dtype=U.dtype)
        
        U_inv = torch.linalg.solve_triangular(U + torch.diag(s), eye, upper=True)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False, unitriangular=True)
        
        W_inv = U_inv @ L_inv @ self.P_inv
        z = x @ W_inv
        logdet = torch.sum( torch.log( torch.abs(s) ), dim=0, keepdim=True )

        return z, -logdet

class MetaBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        cond_channels: int
    ):
        super().__init__()
        final_cond_channels = int( np.ceil(in_channels / 2) ) + cond_channels

        self.l = InvertiblePLU(
            features = in_channels
        )
        
        self.t = nn.Sequential(
            nn.Linear(final_cond_channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, in_channels // 2)
        )
        last_layer = self.t[-1]
        nn.init.zeros_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)


        self.s = nn.Sequential(
            nn.Linear(final_cond_channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, channels), nn.LeakyReLU(), nn.LayerNorm(channels),
            nn.Linear(channels, in_channels // 2)
        )
        last_layer = self.s[-1]
        nn.init.zeros_(last_layer.weight)
        if last_layer.bias is not None:
            nn.init.zeros_(last_layer.bias)
                
    def forward(self, x, y):        
        x, log_det = self.l.forward(x)
        
        x_cond, x_trans = torch.tensor_split(x, 2, dim=1)
        s = self.s( torch.concatenate([x_cond, y], dim=-1) ) 
        t = self.t( torch.concatenate([x_cond, y], dim=-1))
        x_trans = (x_trans - t) * torch.exp(-s)
        x = torch.concatenate((x_cond, x_trans), dim=1)
        
        log_det = log_det -( s ).sum(dim=1)
        
        return x, log_det

    def reverse(self, z, y):
        z_cond, z_trans = torch.tensor_split(z, 2, dim=1)
        s = self.s( torch.concatenate([z_cond, y], dim=-1) ) 
        t = self.t( torch.concatenate([z_cond, y], dim=-1) )
        z_trans = z_trans * torch.exp(s) + t
        z = torch.concatenate((z_cond, z_trans), dim=1)

        z, _ = self.l.reverse(z)
        
        return z
    
class RealNVP(nn.Module):
    def __init__(self, in_channels, channels, cond_channels, n_layers, prior):
        super().__init__()        
        blocks = []
        for i in range(n_layers):
            blocks.append(
                MetaBlock(
                    in_channels,
                    channels,
                    cond_channels,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)
        self.prior = prior

    def forward(self, x, y):
        outputs = []
        log_dets = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            x, log_det = block(x, y)
            log_dets = log_dets + log_det
            outputs.append(x)
        return x, outputs, log_dets

    def reverse(self, x, y, return_sequence = False):
        seq = [x]
        for block in reversed(self.blocks):
            x = block.reverse(x, y)
            seq.append(x)
    
        if not return_sequence:
            return x
        else:
            return seq

class GEncoder(nn.Module):
    def __init__(self, input_size, rep_size):
        super().__init__()
        self.input_size = input_size
        self.rep_size = rep_size

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc_out = nn.Linear(512, rep_size)

        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.norm4 = nn.LayerNorm(512)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        x = self.fc1(g)
        x = self.norm1(x)
        x = F.silu(x)  
        
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.silu(x)

        x = self.fc3(x)
        x = self.norm3(x)
        x = F.silu(x)

        x = self.fc4(x)
        x = self.norm4(x)
        x = F.silu(x)

        x = self.fc_out(x)

        return x
