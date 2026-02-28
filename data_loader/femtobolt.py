# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

import numpy as np
from .base_loader import BaseLoader

class FemtoBolt(BaseLoader):  

    def __init__(self, 
                 data_root, 
                 to_tensor=True,
                 *,
                 color_mask = '/color/color_{:06d}.jpg',
                 depth_mask = '/depth/depth_{:06d}.png',
                 depth_pattern = '/depth/depth_??????.png',
                 image_width = 640,
                 image_height = 480,
                 intri_mat = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),  # sample values
                 depth_ratio = 1.0,
                 ):
        
        super(FemtoBolt, self).__init__(
            data_root, 
            to_tensor, 
            color_mask = color_mask,
            depth_mask = depth_mask,
            depth_pattern = depth_pattern,
            image_width = image_width,
            image_height = image_height,
            intri_mat = intri_mat,
            depth_ratio = depth_ratio,
            )
