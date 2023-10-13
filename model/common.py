from click import style
import torch.nn as nn 
import torch
import logging as logging


class MLP(nn.Module):
    """
    Multilayer perceptron class for information processing in attention blocks

    Param:
        in_features: torch.Tensor - number of input features
        hidden_features: torch.Tensor - number of hidden features
        out_features: torch.Tensor - number of output features
        act_layer: nn.GELU - activation function to activate the weights between linear layers
        drop_rate: torch.Tensor - dropout rate

    Input:
        x: torch.Tensor - input features
    
    Output:
        x: torch.Tensor - input features processed byt the MLP stack of linear layers
    """
    def __init__(self, 
        in_features:torch.Tensor, 
        hidden_features:torch.Tensor,
        out_features:torch.Tensor,
        act_layer=nn.GELU,
        drop_rate:float=torch.tensor(0.0)
    ) -> None:

        super().__init__()
        self.drop_rate = drop_rate
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act_1 = act_layer()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(self.drop_rate)
        

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.fc1(x)
        x = self.act_1(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x



class Permute(nn.Module):
    def __init__(self, dims:float) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x.permute(dims=self.dims)

def drop_path(x:torch.Tensor, drop_prob:float=0.0, training:bool=False):
    """
    Stochastic depth per sample (vertical dropout)
    """
    if drop_prob == 0 or not training:
        return x
    
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1, ) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    output = x.div(keep_prob)*mask
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)