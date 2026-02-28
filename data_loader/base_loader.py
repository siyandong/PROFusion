# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

import numpy as np
import torch
import PIL
from copy import deepcopy
from glob import glob
from tqdm import tqdm
import pose_regression.utils.cropping as cropping
from pose_regression.utils.image import cv2, imread_cv2, ImgNorm
from pose_regression.utils.geometry import depthmap_to_world_and_camera_coordinates

class BaseLoader(object):  # batched acceleration not implemented in this version

    def __init__(self, 
                 data_root, 
                 to_tensor=True, 
                 *,
                 color_mask = '/color/color_{:06d}.jpg',
                 depth_mask = '/depth/depth_{:06d}.png',
                 depth_pattern = '/depth/depth_??????.png',
                 image_width = 640,
                 image_height = 480,
                 intri_mat = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
                 depth_ratio = 1.0,
                ):
        super(BaseLoader, self).__init__()
        self.pr_img_size = 224  # fixed resolution: images will be cropped and resized to 224x224, thus losing the information in the border
        
        self.data_root = data_root
        self.color_mask = color_mask
        self.depth_mask = depth_mask
        self.depth_pattern = depth_pattern
        self.depth_ratio = depth_ratio
        self.image_width = image_width
        self.image_height = image_height
        self.intri_mat = intri_mat

        self.pr_frames, self.of_frames = self.load_frames()

        if to_tensor:
            for img in self.pr_frames:
                img['true_shape'] = torch.tensor(img['true_shape'])
        
    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

        # actual cropping (if necessary) with bilinear interpolation. 
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

        return image, depthmap, intrinsics2

    def _depthmap_to_world_and_camera_coordinates(self, depthmap, camera_intrinsics, camera_pose, dataset):
        return depthmap_to_world_and_camera_coordinates(depthmap, camera_intrinsics, camera_pose, dataset)

    def load_frames(self):

        pr_frames, of_frames = [], []

        n_frames = len(sorted(glob(self.data_root+self.depth_pattern)))
        for fid in tqdm(range(n_frames)):
        
            color_path = self.data_root + self.color_mask.format(fid)
            depth_path = self.data_root + self.depth_mask.format(fid)

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
