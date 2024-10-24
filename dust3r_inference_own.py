#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import gradio
import os
import torch
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import copy
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
import cv2

from dust3r.inference import inference
from dust3r.model import load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import open3d as o3d
import matplotlib.pyplot as pl
import json
from natsort import natsorted

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--views",
        type=int,
        default=3,
        help="number of images to reconstruct the scene",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='own_data',
        help="same name as the folder where own data is placed"
    )
    parser.add_argument(
        "--scan_id",
        type=str,
        default='random',
        help="same name as the folder inside own data is placed"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.025,
        help="subsample points to match same density as prior training"
    )
    parser.add_argument(
        "--normalize_pose",
        type=bool,
        default=True,
        help="normalize pointcloud and camera to be in unit cube"
    )
    return parser


def save_point_cloud_to_ply(points, filename, colors=None):
    num_points = len(points)
    
    data = np.zeros(num_points, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    data['x'] = points[:, 0]
    data['y'] = points[:, 1]
    data['z'] = points[:, 2]
    if colors is not None:
        data['red'] = colors[:, 0]
        data['green'] = colors[:, 1]
        data['blue'] = colors[:, 2]

    vertex = PlyElement.describe(data, 'vertex')
    PlyData([vertex]).write(filename)

def sample_pointcloud(points, colors, target_distance):
    N = points.shape[0]
    
    target_distance_sq = target_distance ** 2
    
    sampled_indices = np.zeros(N, dtype=np.int32)
    sampled_indices[0] = np.random.randint(N)
    
    distances = np.full(N, np.inf)
    n_sampled = 1

    diff = np.zeros_like(points)
    
    while n_sampled < N:
        last_point = points[sampled_indices[n_sampled - 1]]
        
        np.subtract(points, last_point, out=diff)
        current_dist = np.sum(diff * diff, axis=1)
        np.minimum(distances, current_dist, out=distances)
        
        farthest_idx = np.argmax(distances)
        sampled_indices[n_sampled] = farthest_idx
        n_sampled += 1
        
        if n_sampled % 100 == 0 or distances[farthest_idx] < target_distance_sq:
            sampled_points = points[sampled_indices[:n_sampled]]
            tree = cKDTree(sampled_points)
            
            # Query only nearest neighbor distances
            avg_distance = np.mean(tree.query(sampled_points, k=2)[0][:, 1])
            
            if avg_distance < target_distance:
                break
    
    # Return only the valid sampled indices
    final_indices = sampled_indices[:n_sampled]
    return points[final_indices], colors[final_indices]

def get_3D_pc_from_scene(
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
    scale_factor=1.0,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    pts = np.concatenate([p[m] for p, m in zip(pts3d, msk)])
    col = np.concatenate([p[m] for p, m in zip(rgbimg, msk)])
    pointcloud = trimesh.PointCloud(vertices=pts, colors=col)
    return pointcloud

# save json file similar to mipnerf
def save_json(poses, focals, wh, files_path):
    assert len(poses) == len(files_path)
    
    data = {
				"fl_x": focals[0].item(),
				"fl_y": focals[1].item(),
				"cx": wh[0]//2,
				"cy": wh[1]//2,
				"w": wh[0],
				"h": wh[1],
				"frames": [],
			}
    
    # Update data['transforms'] with vis_pose
    for i, pose in enumerate(poses):
        frame = {'file_path': files_path[i], 'transform_matrix': pose.tolist()}
        data['frames'].append(frame)

    # sort according to iamge
    data['frames'] = sorted(data['frames'], key=lambda x: x['file_path'])
    return data

def normalize_pointcloud_and_cameras(pointcloud, camera_positions):
    # Calculate the centroid of the point cloud
    centroid = np.mean(pointcloud, axis=0)
    
    # Translate the point cloud to center it at the origin
    pointcloud_centered = pointcloud - centroid
    
    # Find the maximum extent of the centered point cloud
    max_extent = np.max(np.abs(pointcloud_centered))
    
    # Scale factor to fit the point cloud in a unit cube
    scale_factor = 2.0  / max_extent
    
    # Normalize the point cloud
    pointcloud_normalized = pointcloud_centered * scale_factor
    
    # Apply the same transformation to camera positions
    camera_positions_normalized = (camera_positions - centroid) * scale_factor
    
    return pointcloud_normalized, camera_positions_normalized

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    device = "cuda"
    batch_size = 1
    schedule = "cosine"
    lr = 0.01
    niter = 300

    model = load_model(model_path, device)

    assert args.views == 3
    
    # load images, poses
    scan_id = str(args.scan_id)
    path = os.path.join(f'../data/{args.dataset}/{scan_id}')
    files_path = natsorted(os.listdir(path))

    # use the first two images to reconstruct the scene
    images = load_images(files_path, size=512)       # [384, 512]
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, fx_and_fy=False #PointCloudOptimizer
    )

    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()                                                                                                                                                                                                                                                                                         
    poses = scene.get_im_poses()

    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    
    # save pointcloud path
    save_points_path = f'../data/{args.dataset}/{scan_id}/{str(scan_id)}.ply'    
    pointcloud = get_3D_pc_from_scene(scene, min_conf_thr=10, as_pointcloud=True, mask_sky=True, 
                            clean_depth=True)

    # subsample points
    if args.subsample:
        print('---subsampling pts to match same density of neural points---')
        print('--- might take some time ---')
        pointcloud.vertices, pointcloud.colors = \
            sample_pointcloud(pointcloud.vertices, pointcloud.colors, target_distance=args.subsample)   
    
    # normalize pointcloud and cameras to be in unit cube
    if args.normalize_pose:
        pointcloud.vertices, translation = normalize_pointcloud_and_cameras(pointcloud.vertices, poses[:, :3, 3])
        poses[:, :3, 3] = translation   # updated camera position
    
    # save intrinsic and poses in json format
    data_dict = save_json(poses, focals[0], imgs[0].shape, files_path)
    OUT_PATH = f'../data/{args.dataset}/{scan_id}/{str(scan_id)}.json'   
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(data_dict, outfile, indent=4)
    
    # save pointcloud
    pointcloud.export(save_points_path)