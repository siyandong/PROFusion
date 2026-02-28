# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions about images (loading/converting...)
# --------------------------------------------------------
import os
import torch
import numpy as np
import PIL.Image
from tqdm import tqdm
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

from .geometry import depthmap_to_camera_coordinates
import json

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def rgb(ftensor, true_shape=None):
    if isinstance(ftensor, list):
        return [rgb(x, true_shape=true_shape) for x in ftensor]
    if isinstance(ftensor, torch.Tensor):
        ftensor = ftensor.detach().cpu().numpy()  # H,W,3
    if ftensor.ndim == 3 and ftensor.shape[0] == 3:
        ftensor = ftensor.transpose(1, 2, 0)
    elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
        ftensor = ftensor.transpose(0, 2, 3, 1)
    if true_shape is not None:
        H, W = true_shape
        ftensor = ftensor[:H, :W]
    if ftensor.dtype == np.uint8:
        img = np.float32(ftensor) / 255
    else:
        img = (ftensor * 0.5) + 0.5
    return img.clip(min=0, max=1)


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(folder_or_list, size, square_ok=False, 
                verbose=1, img_num=0, img_freq=0, 
                postfix=None, start_idx=0):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose > 0:
            print(f'>> Loading images from {folder_or_list}')
        img_names = [name for name in os.listdir(folder_or_list) if not "depth" in name]
        if postfix is not None:
            img_names = [name for name in img_names if name.endswith(postfix)]
        root, folder_content = folder_or_list, img_names
        
    elif isinstance(folder_or_list, list):
        if verbose > 0:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')
   
    # sort images by number in name
    len_postfix = len(postfix) if postfix is not None \
        else len(folder_content[0]) - folder_content[0].rfind('.')

    img_numbers = []
    for name in folder_content:
        dot_index = len(name) - len_postfix
        number_start = 0
        for i in range(dot_index-1, 0, -1):
            if not name[i].isdigit():
                number_start = i + 1
                break
        img_numbers.append(float(name[number_start:dot_index]))
    folder_content = [x for _, x in sorted(zip(img_numbers, folder_content))]

    if start_idx > 0:
        folder_content = folder_content[start_idx:]
    if(img_freq > 0):
        folder_content = folder_content[::img_freq]
    if(img_num > 0):
        folder_content = folder_content[:img_num]
        
    # print(root, folder_content)

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    if verbose > 0:
        folder_content = tqdm(folder_content, desc='Loading images')
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        if verbose > 1:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs)), label=path))
            
    assert imgs, 'no images foud at '+ root 
    if verbose > 0:
        print(f' ({len(imgs)} images loaded)')
    return imgs



def crop_and_resize(image, depthmap, intrinsics, long_size, rng=None, info=None, use_crop=False):
    """ This function:
        1. 将图片crop,使得其principal point真正落在中间
        2. 根据图片横竖确定target resolution的横竖
    """
    import slam3r.datasets.utils.cropping as cropping
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
        
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    if(use_crop):
        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
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
    scale = long_size / max(W, H)
    
    # high-quality Lanczos down-scaling
    target_resolution = np.array([W, H]) * scale

    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution)

    return image, depthmap, intrinsics


def load_scannetpp_images_pts3dcam(folder_or_list, size, square_ok=False, verbose=True, img_num=0, img_freq=0):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    if(img_freq > 0):
        folder_content = folder_content[1000::img_freq]
    if(img_num > 0):
        folder_content = folder_content[:img_num]
        
    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []

    intrinsic_path = os.path.join(os.path.dirname(root), 'pose_intrinsic_imu.json')
    with open(intrinsic_path, 'r') as f:
        info = json.load(f)
    
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img_path = os.path.join(root, path)
        img = exif_transpose(PIL.Image.open(img_path)).convert('RGB')
        W1, H1 = img.size
    
        depth_path = img_path.replace('.jpg', '.png').replace('rgb','depth')
        depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000.
        """
        img and depth has different convention about shape
        """
        # print(img.size, depthmap.shape)
        depthmap = cv2.resize(depthmap, (W1,H1), interpolation=cv2.INTER_CUBIC)
        # print(img.size, depthmap.shape)
        img_id = os.path.basename(img_path)[:-4]
        intrinsics = np.array(info[img_id]['intrinsic'])
        # print(img, depthmap, intrinsics)
        img, depthmap, intrinsics = crop_and_resize(img, depthmap, intrinsics, size)
        # print(img, depthmap, intrinsics)
        pts3d_cam, mask = depthmap_to_camera_coordinates(depthmap, intrinsics)
        pts3d_cam = pts3d_cam * mask[..., None] 
        # print(pts3d_cam.shape)
        valid_mask = np.isfinite(pts3d_cam).all(axis=-1)
        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        
        imgs.append(dict(img=ImgNorm(img)[None], 
                         true_shape=np.int32([img.size[::-1]]), 
                         idx=len(imgs), 
                         instance=str(len(imgs)),
                         pts3d_cam=pts3d_cam[None],
                         valid_mask=valid_mask[None]
                         ))
        # break
    
    assert imgs, 'no images foud at '+root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs            
