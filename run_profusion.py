# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

import numpy as np
import sys
import os

# PROFusion modules
from pose_regression.run import load_pose_regression_model, run_relative_pose_regression
sys.path.insert(0, "./build")
import optimization_fusion as OFP
# data loaders
from data_loader import FemtoBolt, FastCaMo, ETH3D

from tqdm import tqdm

def main():

    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Wrong args, usage: python run_profusion.py <data_config>")
        sys.exit(1)

    data_config = OFP.DataConfiguration(sys.argv[1]) 
    controller_config = OFP.ControllerConfiguration("configs/controller_config/default.yaml")
    if len(sys.argv) == 3:
        controller_config = OFP.ControllerConfiguration(sys.argv[2])
    result_path = data_config.output_folder
    os.makedirs(result_path, exist_ok=True)

    print("\n# Loading PROFusion...")
    PR = load_pose_regression_model()
    pipeline = OFP.OptFus(data_config, controller_config)

    print("\n# Loading RGB-D Frames...")
    if 'femtobolt' in data_config.data_root:
        dataset_cls = FemtoBolt
    elif 'fastcamo' in data_config.data_root:
        dataset_cls = FastCaMo
    elif 'eth3d' in data_config.data_root:
        dataset_cls = ETH3D
    else:
        raise ValueError("Invalid dataset format: {}".format(data_config.data_root))
    dataset = dataset_cls(data_root=data_config.data_root,
                          image_width=data_config.image_width,
                          image_height=data_config.image_height,
                          intri_mat=np.array([[data_config.fx, 0.0, data_config.cx], [0.0, data_config.fy, data_config.cy], [0.0, 0.0, 1.0],], dtype=np.float32),
                          depth_ratio=data_config.depth_ratio, )

    print("\n# Pose Regression")
    relposes = run_relative_pose_regression(PR, dataset.pr_frames)
    for relpose in relposes:
        relpose[:3, 3] *= 1000.0  # convert unit from m to mm
    initial_poses = relposes
    def get_relpose(initial_poses, frame_id):
        if frame_id < 0 or frame_id > len(initial_poses):
            return None
        return initial_poses[frame_id]

    print("\n# Optimization and Fusion")
    n_imgs = 0
    success_count = 0
    for fid in tqdm(range(len(dataset.of_frames)), desc="optimizing poses and fusing frames"):
        color_img = dataset.of_frames[fid]["color"]
        depth_map_float = dataset.of_frames[fid]["depth"].astype(np.float32)
        rel_pose = get_relpose(initial_poses, n_imgs - 1)
        try:
            if rel_pose is not None:
                success = pipeline.process_frame(depth_map_float, color_img, rel_pose)
            else:
                success = pipeline.process_frame(depth_map_float, color_img)
            if success:
                success_count += 1
            else:
                print("frame {} failed".format(fid))      
        except Exception as e:
            print("frame {} error {}".format(fid, e))
        n_imgs += 1
    
    if controller_config.save_trajectory:
        print("\n# Saving camera poses")
        poses_file = result_path + "/camera_poses.txt"
        pipeline.save_poses(poses_file)
        print("saved to {}".format(poses_file)) 

    if controller_config.save_scene:
        print("\n# Saving scene points")
        points = pipeline.extract_pointcloud()
        print("in total {} points".format(points.num_points))
        ply_file = result_path + "/scene_points.ply"
        pipeline.export_ply(ply_file, points)
        print("saved to {}".format(ply_file))

if __name__ == "__main__":
    main()
