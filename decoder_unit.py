import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from einops import rearrange
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import einops


class DecoderUNit(nn.Module):
  def __init__(self, inchannels, outchannels, size):
    super().__init__()
    self.up = nn.Upsample(size = size)
    self.conv1 = nn.Sequential(
        nn.Conv2d(inchannels[0], inchannels[0] , kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(inchannels[0]),
        nn.ReLU(),
    )
        # nn.Upsample(size = size),
        # nn.Conv2d(256 + inchannels, outchannels),
        # nn.BatchNorm2d(outchannels),
        # nn.Relu(),
    self.conv2 = nn.Sequential(
        nn.Conv2d(inchannels[1], outchannels, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(outchannels),
        nn.ReLU()      
    )
    # pass
  def forward(self, x, en = None, patch = None):
    if en is not None:
      # skip = self.skip(en, patch)
      # skip = rearrange(en, 'b (p1 p2) c -> b c p1 p2', p1 = patch, p2 = patch)
      x = x + en
      shortcut = x.clone()
      x = self.conv1(x)
      x = x + shortcut
    if en is not None:
      x = torch.cat([x,en], dim = 1 )
    x = self.up(x)
    x = self.conv2(x)

    return x
