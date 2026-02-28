# Pose regression head from Reloc3r

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from pdb import set_trace as bb

# code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L172'
class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

# parts of the code adapted from 'https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py#L193'
class PoseHead(nn.Module):
    """ 
    pose regression head
    """
    def __init__(self, 
                 net, 
                 num_resconv_block=2,
                 rot_representation='9D'):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.num_resconv_block = num_resconv_block
        self.rot_representation = rot_representation  

        output_dim = 4*self.patch_size**2

        self.proj = nn.Linear(net.dec_embed_dim, output_dim)
        self.res_conv = nn.ModuleList([copy.deepcopy(ResConvBlock(output_dim, output_dim)) 
            for _ in range(self.num_resconv_block)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim,output_dim),
            nn.ReLU(),
            nn.Linear(output_dim,output_dim),
            nn.ReLU()
            )
        self.fc_t = nn.Linear(output_dim, 3)
        if self.rot_representation=='9D':
            self.fc_rot = nn.Linear(output_dim, 9)
        else:
            self.fc_rot = nn.Linear(output_dim, 6)
        
    def svd_orthogonalize(self, m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # if m.shape[0] == 4:
            # print(torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),u.transpose(-2, -1))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        # if m.shape[0] == 4:
        #     print(r)
        return r

    def rotation_6d_to_matrix(self, d6):  # code from pytorch3d
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)

        Returns:
            batch of rotation matrices of size (*, 3, 3)

        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)
    
    def convert_pose_to_4x4(self, B, out_r, out_t, device):
        if self.rot_representation=='9D':
            out_r = self.svd_orthogonalize(out_r)  # [N,3,3]
        else:
            out_r = self.rotation_6d_to_matrix(out_r)
        pose = torch.zeros((B, 4, 4), device=device)
        pose[:, :3, :3] = out_r
        pose[:, :3, 3] = out_t
        pose[:, 3, 3] = 1.
        return pose

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        for i in range(self.num_resconv_block):
            feat = self.res_conv[i](feat)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        feat = self.more_mlps(feat)  # [B, D_]
        out_t = self.fc_t(feat)  # [B,3]
        out_r = self.fc_rot(feat)  # [B,9]
        pose = self.convert_pose_to_4x4(B, out_r, out_t, tokens.device)
        res = {"pose": pose}

        return res

