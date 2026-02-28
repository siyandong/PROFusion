# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

import torch
import torch.nn as nn
from .modules.base_model import Multiview3D
from .modules.pose_head import PoseHead
from .utils.device import MyNvtxRange
from .utils.misc import transpose_to_landscape
from pdb import set_trace as bb

# Metric-aware relative camera pose regression model
class PROFusionPoseRegression(Multiview3D):  

    def __init__(self, **args):
        super().__init__(**args)
        self.dec_embed_dim = self.decoder_embed.out_features
        self.void_pe_token = nn.Parameter(torch.randn(1,1,self.dec_embed_dim), requires_grad=True)
        self.ponit_embedder = nn.Conv2d(3, self.dec_embed_dim, kernel_size=self.patch_size, stride=self.patch_size)        
        self._set_downstream_head()

    def _set_downstream_head(self):
        # allocate heads: only one head is needed to regress relative pose with source tokens
        self.pose_downstream_head2 = PoseHead(net=self)
        self.pose_head2 = transpose_to_landscape(self.pose_downstream_head2)
        
    def get_pe(self, views, ref_ids):
        """embed 3D points with a single conv layer"""
        pes = []
        for id, view in enumerate(views):
            if id in ref_ids:
                pos = view['pts3d_ref']
                # pos = view['pts3d_local']  # assuming only one ref view, robust to unstable camera motions
            else:
                pos = view['pts3d_cam']
                # pos = view['pts3d_local']
                
            if pos.shape[-1] == 3:
                pos = pos.permute(0, 3, 1, 2)
                
            pts_embedding = self.ponit_embedder(pos).permute(0,2,3,1).reshape(pos.shape[0], -1, self.dec_embed_dim) # (B, S, D)
            if 'patch_mask' in view:
                patch_mask = view['patch_mask'].reshape(pos.shape[0], -1, 1) # (B, S, 1)
                pts_embedding = pts_embedding*(~patch_mask) + self.void_pe_token*patch_mask
                
            pes.append(pts_embedding)
        
        return pes
    
    def _decode_multiview(self, 
                          ref_feats:torch.Tensor, src_feats:torch.Tensor, 
                          ref_poses:torch.Tensor, src_poses:torch.Tensor, 
                          ref_pes:torch.Tensor|None, src_pes:torch.Tensor|None):
        """exchange information between reference and source views in the decoder

        About naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.

        Args:
            ref_feats (R, B, S, D_enc): img tokens of reference views 
            src_feats (V-R, B, S, D_enc): img tokens of source views
            ref_poses (R, B, S, 2): positions of tokens of reference views
            src_poses (V-R, B, S, 2): positions of tokens of source views
            ref_pes (R, B, S, D_dec): pointmap tokens of reference views
            src_pes:(V-R, B, S, D_dec): pointmap tokens of source views
        
        Returns:
            final_refs: list of R*(B, S, D_dec)
            final_srcs: list of (V-R)*(B, S, D_dec)
        """
        # R: number of reference views
        # V: total number of reference and source views
        # S: number of tokens
        num_ref = ref_feats.shape[0]
        num_src = src_feats.shape[0]
        
        final_refs = [ref_feats]  # before projection
        final_srcs = [src_feats]
        # project to decoder dim
        final_refs.append(self.decoder_embed(ref_feats)) 
        final_srcs.append(self.decoder_embed(src_feats))
        
        # define how each views interact with each other
        # here we use a simple way: ref views and src views exchange information bidirectionally
        # for more detail, please refer to the class MultiviewDecoderBlock_max in blocks/multiview_blocks.py
        src_rel_ids_d = torch.arange(num_ref, device=final_refs[0].device, dtype=torch.long)
        src_rel_ids_d = src_rel_ids_d[None].expand(src_feats.shape[0], -1).reshape(-1) # (V-R * R)
        ref_rel_ids_d = torch.arange(num_src, device=final_refs[0].device, dtype=torch.long)
        ref_rel_ids_d = ref_rel_ids_d[None].expand(ref_feats.shape[0], -1).reshape(-1) # (R * V-R)
        
        for i in range(self.dec_depth):
            # (R, B, S, D),  (V-R, B, S, D)
            # add pointmap tokens if available(used in Local2WorldModel)
            ref_inputs = final_refs[-1] + ref_pes if ref_pes is not None else final_refs[-1]
            src_inputs = final_srcs[-1] + src_pes if src_pes is not None else final_srcs[-1]

            ref_blk:MultiviewDecoderBlock_max = self.mv_dec_blocks1[i]
            src_blk = self.mv_dec_blocks2[i]
            # reference image side
            ref_outputs = ref_blk(ref_inputs, src_inputs, 
                                     ref_poses, src_poses, 
                                     # ref_rel_ids_d, num_src) # (R, B, S, D)
                                     ref_rel_ids_d, num_src, disable_cross_attention=True) # (R, B, S, D)
            # source image side
            src_outputs = src_blk(src_inputs, ref_inputs, 
                                     src_poses, ref_poses, 
                                     src_rel_ids_d, num_ref) # (V-R, B, S, D)
            # store the result
            final_refs.append(ref_outputs)
            final_srcs.append(src_outputs)

        # normalize last output
        del final_srcs[1]  # duplicate with final_output[0]
        del final_refs[1]
        final_refs[-1] = self.dec_norm(final_refs[-1])  #(R, B, S, D)
        final_srcs[-1] = self.dec_norm(final_srcs[-1])

        for i in range(len(final_refs)):
            R, B, S, D = final_refs[i].shape
            assert R == num_ref
            final_refs[i] = final_refs[i].reshape(R*B, S, D)  #(R*B, S, D)
            final_srcs[i] = final_srcs[i].reshape(num_src*B, S, D)  #((V-R)*B, S, D/D')

        return final_refs, final_srcs  #list: [depth*(R*B, S, D/D')], [depth*((V-R)*B, S, D/D')]

    def forward(self, views:list, ref_ids = 0, return_pose=True):
        """ 
        naming convention:
            reference views: views that define the coordinate system.
            source views: views that need to be transformed to the coordinate system of the reference views.
        
        Args:
            views: list of dictionaries, each containing:
                    - 'img': input image tensor (B, 3, H, W) or 'img_tokens': image tokens (B, S, D)
                    - 'true_shape': true shape of the input image (B, 2)
                    - 'pts3d_ref' (reference view only): 3D pointmaps in the ref coordinate system (B, H, W, 3)
                    - 'pts3d_cam' (source view only): 3D pointmaps in the camera coordinate system (B, H, W, 3)
            ref_ids: indexes of the reference views in the input view list
        """
        # decide which views are reference views and which are source views
        if isinstance(ref_ids, int):
            ref_ids = [ref_ids]
        for ref_id in ref_ids:
            assert ref_id < len(views) and ref_id >= 0
        src_ids = [i for i in range(len(views)) if i not in ref_ids]            

        # #feat: B x S x D  pos: B x S x 2
        with MyNvtxRange('encode'):
            shapes, enc_feats, poses = self._encode_multiview(views)
            pes = self.get_pe(views, ref_ids=ref_ids)
        
        # select and stacck up ref and src elements
        ref_feats, src_feats = self.split_stack_ref_src(enc_feats, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_poses, src_poses = self.split_stack_ref_src(poses, ref_ids, src_ids)  # (R, B, S, 2), (V-R, B, S, 2)
        ref_pes, src_pes = self.split_stack_ref_src(pes, ref_ids, src_ids) # (R, B, S, D), (V-R, B, S, D)
        ref_shapes, src_shapes = self.split_stack_ref_src(shapes, ref_ids, src_ids) # (R, B, 2), (V-R, B, 2)
        
        # combine all ref images into object-centric representation
        with MyNvtxRange('decode'):
            dec_feats_ref, dec_feats_src = self._decode_multiview(ref_feats, src_feats, 
                                                                  ref_poses, src_poses, 
                                                                  ref_pes, src_pes)
        
        with MyNvtxRange('head'):
            with torch.amp.autocast('cuda', enabled=False):  # currently the pose head does not support amp
                pose2 = self.pose_head2([tok.float() for tok in dec_feats_src], src_shapes.reshape(-1,2))['pose']
        
        # split the results back to each view
        results = [] 
        B = pose2.shape[0] // len(ref_ids)
        for id in range(len(views)):
            res = {}
            if id in ref_ids:
                pass
            else:
                rel_id = src_ids.index(id)
                res['pose'] = pose2[rel_id*B:(rel_id+1)*B] 
            results.append(res)
        return results
