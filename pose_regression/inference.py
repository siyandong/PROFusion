# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

import torch
import numpy as np
from time import time
from pdb import set_trace as bb

def to_device(view, device='cuda'):  # dict of tensor
    for name in 'img pts3d_cam pts3d_ref true_shape img_tokens'.split():
        if name in view:
            view[name] = view[name].to(device)

@torch.no_grad()
def rpr_inference(raw_views, rpr_model, ref_ids, 
                  masks=None,
                  normalize=False, 
                  device='cuda'):
    # construct new input to avoid modifying the raw views
    input_views = [dict(
                        img_tokens=view['img_tokens'], 
                        true_shape=view['true_shape'],
                        img_pos=view['img_pos'],
                        # camera_pose=view['camera_pose'],
                        ) 
                   for view in raw_views]

    for view in input_views:
        to_device(view, device=device) 

    for id, view in enumerate(raw_views):            
        if id in ref_ids:
            pts_world = view['pts3d_ref']
            if masks is not None:
                pts_world = pts_world*(masks[id].float())
            input_views[id]['pts3d_ref'] = pts_world
        else:
            input_views[id]['pts3d_cam'] = raw_views[id]['pts3d_cam']
            if masks is not None:
                input_views[id]['pts3d_cam'] = input_views[id]['pts3d_cam']*(masks[id].float())
    
    for view in input_views:
        to_device(view, device=device)   

    with torch.no_grad():
        output = rpr_model(input_views, ref_ids=ref_ids)

    return output
