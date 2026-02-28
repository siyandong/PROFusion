# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

import numpy as np
from .base_loader import BaseLoader
from copy import deepcopy
from glob import glob
from tqdm import tqdm
from pose_regression.utils.image import cv2, imread_cv2, ImgNorm

class ETH3D(BaseLoader):  

    def __init__(self, 
                 data_root, 
                 to_tensor=True,
                 *,
                 depth_pattern = '/depth/*.png',
                 image_width = 739,
                 image_height = 458,
                 intri_mat = np.array([[726.2874, 0., 354.6497], [0., 726.2874, 186.4657], [0., 0., 1.]], dtype=np.float32),
                 depth_ratio = 5.0,
                 ):
        
        super(ETH3D, self).__init__(
            data_root, 
            to_tensor,
            depth_pattern = depth_pattern,
            image_width = image_width,
            image_height = image_height,
            intri_mat = intri_mat,
            depth_ratio = depth_ratio,
            )

    def load_frames(self):

        pr_frames, of_frames = [], []

        depth_paths = sorted(glob(self.data_root+self.depth_pattern))
        for depth_path in tqdm(depth_paths):
            color_path = depth_path.replace('depth', 'rgb')

            color = imread_cv2(color_path)
            color = cv2.resize(color, (self.image_width, self.image_height))  # align with depth
            depth = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED) / self.depth_ratio

            of_frames.append({'color': deepcopy(np.ascontiguousarray(color[:,:,[2,1,0]])), 'depth': deepcopy(depth)})

            depth = depth.astype(np.float32) / 1000.  # meters used in pose regression

            H1, W1, _ = color.shape
            _H1, _W1 = depth.shape
            assert H1==_H1 and W1==_W1

            color, depth, intrinsics = self._crop_resize_if_necessary(
                color, depth, deepcopy(self.intri_mat), (self.pr_img_size,self.pr_img_size), rng=None, info=None)

            W2, H2 = color.size

            _, pts3d_cam, valid_mask = self._depthmap_to_world_and_camera_coordinates(
                depthmap=depth, 
                camera_intrinsics=intrinsics.astype(np.float32), 
                camera_pose=np.identity(4).astype(np.float32),  # identity pose
                dataset='FemtoBlot')

            pr_frames.append(dict(
                img=ImgNorm(color)[None], 
                true_shape=np.int32([color.size[::-1]]),
                idx=len(pr_frames), 
                instance=str(len(pr_frames)), 
                label=color_path, 
                label_d=depth_path, 
                camera_intrinsics=intrinsics.astype(np.float32),
                camera_pose=None,
                pts3d_local=pts3d_cam,
                valid_mask=valid_mask
                ))
        return pr_frames, of_frames
