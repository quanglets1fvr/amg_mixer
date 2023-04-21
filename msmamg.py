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


def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1]) 
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


# MFI
class GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nn.Module):
    """Get gating weights for cross-gating MLP block."""
    def __init__(self,nIn:int,Nout:int,H_size:int=128,W_size:int=128,input_proj_factor:int=2,dropout_rate:float=0.0,use_bias:bool=True,train_size:int=512):
        super(GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid, self).__init__()
        
        self.H = H_size
        self.W = W_size
        self.IN = nIn
        self.OUT = Nout
        if train_size == 512:
            self.grid_size = [[8, 8], [4, 4], [2, 2]]
        else:
            self.grid_size = [[6, 6], [3, 3], [2, 2]]

        self.block_size = [[int(H_size / l[0]), int(W_size / l[1])] for l in self.grid_size]
        self.input_proj_factor = input_proj_factor
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNorm = nn.LayerNorm(self.IN)
        self.Linear_end = nn.Linear(self.IN,self.OUT)
        self.Gelu = nn.GELU()
        self.Linear_grid_MLP_1 = nn.Linear((self.grid_size[0][0]*self.grid_size[0][1]),(self.grid_size[0][0]*self.grid_size[0][1]),bias=use_bias)

        self.Linear_Block_MLP_1 = nn.Linear((self.block_size[0][0]*self.block_size[0][1]),(self.block_size[0][0]*self.block_size[0][1]),bias=use_bias)

        self.Linear_grid_MLP_2 = nn.Linear((self.grid_size[1][0] * self.grid_size[1][1]),
                                           (self.grid_size[1][0] * self.grid_size[1][1]), bias=use_bias)

        self.Linear_Block_MLP_2 = nn.Linear((self.block_size[1][0] * self.block_size[1][1]),
                                            (self.block_size[1][0] * self.block_size[1][1]), bias=use_bias)

        self.Linear_grid_MLP_3 = nn.Linear((self.grid_size[2][0] * self.grid_size[2][1]),
                                           (self.grid_size[2][0] * self.grid_size[2][1]), bias=use_bias)

        self.Linear_Block_MLP_3 = nn.Linear((self.block_size[2][0] * self.block_size[2][1]),
                                            (self.block_size[2][0] * self.block_size[2][1]), bias=use_bias)

    def forward(self, x): 
        n, h, w,num_channels = x.shape
        
        x = self.LayerNorm(x.float()) 
        x = self.Gelu(x)

       
        gh1, gw1 = self.grid_size[0]
        fh1, fw1 = h // gh1, w // gw1
        u1 = block_images_einops(x, patch_size=(fh1, fw1))
        u1 = u1.permute(0,3,2,1)

        u1 = self.Linear_grid_MLP_1(u1)
        u1 = u1.permute(0,3,2,1)
        u1 = unblock_images_einops(u1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        fh1, fw1 = self.block_size[0]
        gh1, gw1 = h // fh1, w // fw1
        v1 = block_images_einops(u1, patch_size=(fh1, fw1))
        v1 = v1.permute(0, 1, 3, 2)
        v1 = self.Linear_Block_MLP_1(v1)
        v1 = v1.permute(0, 1, 3, 2)
        v1 = unblock_images_einops(v1, grid_size=(gh1, gw1), patch_size=(fh1, fw1))

        gh2, gw2 = self.grid_size[1]
        fh2, fw2 = h // gh2, w // gw2
        u2 = block_images_einops(v1, patch_size=(fh2, fw2)) 
        u2 = u2.permute(0, 3, 2, 1)

        u2 = self.Linear_grid_MLP_2(u2)
        u2 = u2.permute(0, 3, 2, 1)
        u2 = unblock_images_einops(u2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        fh2, fw2 = self.block_size[1]
        gh2, gw2 = h // fh2, w // fw2
        v2 = block_images_einops(u2, patch_size=(fh2, fw2))
        v2 = v2.permute(0, 1, 3, 2)
        v2 = self.Linear_Block_MLP_2(v2)
        v2 = v2.permute(0, 1, 3, 2)
        v2 = unblock_images_einops(v2, grid_size=(gh2, gw2), patch_size=(fh2, fw2))

        gh3, gw3 = self.grid_size[2]
        fh3, fw3 = h // gh3, w // gw3
        u3 = block_images_einops(v2, patch_size=(fh3, fw3))  
        u3 = u3.permute(0, 3, 2, 1)

        u3 = self.Linear_grid_MLP_3(u3)
        u3 = u3.permute(0, 3, 2, 1)
        u3 = unblock_images_einops(u3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        fh3, fw3 = self.block_size[2]
        gh3, gw3 = h // fh3, w // fw3
        v3 = block_images_einops(u3, patch_size=(fh3, fw3))
        v3 = v3.permute(0, 1, 3, 2)
        v3 = self.Linear_Block_MLP_3(v3)
        v3 = v3.permute(0, 1, 3, 2)
        v3 = unblock_images_einops(v3, grid_size=(gh3, gw3), patch_size=(fh3, fw3))

        x = self.Linear_end(v3)
        x = self.dropout(x)
        return x


class conv_T_y_2_x(nn.Module):
    """ Unified y Dimensional to x """
    def __init__(self,y_nIn,x_nOut):
        super(conv_T_y_2_x, self).__init__()
        self.x_c = x_nOut
        self.y_c = y_nIn
        self.convT = nn.ConvTranspose2d(in_channels=self.y_c, out_channels=self.x_c, kernel_size=(3,3),
                                        stride=(2, 2))
    def forward(self,x,y):
       
        y = self.convT(y)
        _, _, h, w, = x.shape
        y = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        return y



class MSMAMG(nn.Module):
    """Cross-gating MLP block."""
    def __init__(self,x_in:int,y_in:int,out_features:int,patch_size,block_size,grid_size,dropout_rate:float=0.0,input_proj_factor:int=2,upsample_y:bool=True,use_bias:bool=True, train_size:int=512):
        super(MSMAMG, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(out_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, groups = out_features, bias=False),
            nn.Dropout(p=0.1), # save load thi bo Dropout
            nn.BatchNorm2d(out_features),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_features, out_features, kernel_size=1)
        )
        self.IN_x = x_in
        self.IN_y = y_in
        self._h = patch_size[0]
        self._w = patch_size[1]
        self.features = out_features
        self.block_size=block_size
        self.grid_size = grid_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.Conv1X1_x = nn.Conv2d(self.IN_x,self.features,(1,1))
        self.Conv1X1_y = nn.Conv2d(self.IN_x,self.features,(1,1))
        self.LayerNorm_x = nn.LayerNorm(self.features)
        self.LayerNorm_y = nn.LayerNorm(self.features)
        self.Linear_x = nn.Linear(self.features,self.features,bias=use_bias)
        self.Linear_y = nn.Linear(self.features,self.features,bias=use_bias)
        self.Gelu_x = nn.GELU()
        self.Gelu_y = nn.GELU()
        self.Linear_x_end = nn.Linear(self.features,self.features,bias=use_bias)
        self.Linear_y_end = nn.Linear(self.features,self.features,bias=use_bias)
        self.dropout_x = nn.Dropout(self.dropout_rate)
        self.dropout_y = nn.Dropout(self.dropout_rate)

        self.ConvT = conv_T_y_2_x(self.IN_y,self.IN_x)
        self.fun_gx = GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

        self.fun_gy = GetSpatialGatingWeights_2D_Multi_Scale_Cascade_Grid(nIn=self.features, Nout=self.features, H_size=self._h, W_size=self._w,
                                                 input_proj_factor=2, dropout_rate=dropout_rate, use_bias=True, train_size=train_size)

    def forward(self, x):
    

        x = self.Conv1X1_x(x)
        # y = self.Conv1X1_y(y)
        # assert y.shape == x.shape
        x = x.permute(0, 2, 3, 1)  # n x h x w x c
        # y = y.permute(0, 2, 3, 1)
        shortcut_x = x
        # shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.Linear_x(x)
        x = self.Gelu_x(x)

        gx = self.fun_gx(x)
        
        x = self.Linear_y_end(x)
        x = self.dropout_x(x)
        x = x  + shortcut_x  
        x = x.permute(0, 3, 1, 2)  # n x h x w x c --> n x c x h x w
        # y = y.permute(0, 3, 1, 2)
        # logit = torch.cat([x,y], dim = 1)
        x = self.conv(x)
        return x


