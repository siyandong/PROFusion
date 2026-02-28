# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

from .model import PROFusionPoseRegression
from .inference import * 
from tqdm import tqdm

def load_pose_regression_model():
    PR = PROFusionPoseRegression.from_pretrained('siyan824/profusion_pr') 
    # PR = PROFusionPoseRegression()
    # PR.load_state_dict(torch.load('checkpoints/profusion_pr.pth', weights_only=False), strict=False)
    PR.to('cuda')
    PR.eval()
    return PR

@torch.no_grad()
def get_img_tokens(views, model):
    res_shapes, res_feats, res_poses = model._encode_multiview(views, 
                                                               view_batchsize=10, 
                                                               normalize=False,
                                                               silent=False)
    return res_shapes, res_feats, res_poses

def run_relative_pose_regression(PR, pr_frames):

    data_views = pr_frames
    num_views = len(data_views)
    for i in range(num_views):
        if len(data_views[i]['img'].shape)==4 and data_views[i]['img'].shape[0]==1:
            data_views[i]['img'] = data_views[i]['img'][0]
    if 'valid_mask' not in data_views[0]:
        valid_masks = None
    else:
        valid_masks = [view['valid_mask'] for view in data_views]
    
    # preprocess data for extracting their color tokens with encoder
    for view in data_views:
        view['img'] = view['img'][None]
        view['true_shape'] = view['true_shape'][None]  
        for key in ['valid_mask', 'pts3d_cam', 'pts3d']:
            if key in view:
                del view[key]
        to_device(view, device='cuda')

    # color image encoding
    _, res_feats, res_poses = get_img_tokens(data_views, PR)  

    # re-organize input views for the following inference, keep necessary attributes only
    input_views = []
    for i in range(num_views):
        view = dict(label=data_views[i]['label'],
                    img_tokens=res_feats[i], 
                    true_shape=data_views[i]['true_shape'], 
                    img_pos=res_poses[i],
                    )
        if 'pts3d_local' in data_views[i] and 'camera_pose' in data_views[i]:
            view['pts3d_local'] = data_views[i]['pts3d_local']
            # view['camera_pose'] = data_views[i]['camera_pose']
        input_views.append(view)

    # relative pose regression frame-by-frame, batched acceleration not implemented in this version
    relposes = []
    c2w_tmp = [np.identity(4)]
    for view_id in tqdm(range(len(input_views)-1), desc="regressing poses"):  

        view_ref = input_views[view_id]
        view_q = input_views[view_id+1]
        
        view_ref['pts3d_ref'] = torch.tensor(view_ref['pts3d_local'])[None]
        view_q['pts3d_cam'] = torch.tensor(view_q['pts3d_local'])[None]

        output = rpr_inference(raw_views=[view_ref, view_q], ref_ids=[0], rpr_model=PR, device='cuda', normalize=False)
        q2ref_pred = output[-1]['pose'].squeeze().detach().cpu().numpy()

        relposes.append(q2ref_pred)
        c2w_tmp.append(q2ref_pred)

    return relposes
