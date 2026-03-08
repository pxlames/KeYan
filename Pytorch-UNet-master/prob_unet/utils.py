import torch
import torch.nn as nn


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        truncated_normal_(module.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.orthogonal_(module.weight)
        truncated_normal_(module.bias, mean=0, std=0.001)


def l2_regularisation(module):
    l2_reg = None
    for param in module.parameters():
        l2_reg = param.norm(2) if l2_reg is None else l2_reg + param.norm(2)
    return l2_reg

