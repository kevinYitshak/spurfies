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
import tempfile
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import plyfile
from plyfile import PlyData, PlyElement
import cv2

from dust3r.inference import inference
from dust3r.model import load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
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
        default='dtu',
        help="possible datasets 'dtu' or 'mipnerf'"
    )
    parser.add_argument(
        "--scan_id",
        type=str,
        default='24',
        help="dtu: [21,24,34,37,38,40,82,106,110,114,118]; mipnerf: [garden, stump]"
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


def _convert_scene_output_to_glb(
    outdir,
    scene_name,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        # save_point_cloud_to_ply(pts, '/BS/local_diffusion/work/dust3r/estimation/21.ply', col*255)
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    rot = np.eye(4)
    type = ""
    if as_pointcloud:
        type = "pointcloud"
    else:
        type = "mesh"
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, f"{scene_name}_{type}.glb")
    print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(
    outdir,
    scene_name,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
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
    return _convert_scene_output_to_glb(
        outdir,
        scene_name,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
    )

import numpy as np
from scipy.spatial import cKDTree
'''
def sample_pointcloud(points, colors, target_distance):
    N = points.shape[0]
    sampled_indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)
    
    while True:
        last_point = points[sampled_indices[-1]]
        dist = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dist)
        
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        
        if len(sampled_indices) > 1:
            sampled_points = points[sampled_indices]
            tree = cKDTree(sampled_points)
            avg_distance = np.mean(tree.query(sampled_points, k=2)[0][:, 1])
            
            if avg_distance < target_distance:
                break
    
    return points[np.array(sampled_indices[:-1])], colors[np.array(sampled_indices[:-1])]
'''
import numpy as np
from scipy.spatial import cKDTree

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
    outdir,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
    scale_factor=1.0,
    subsample = 0.02
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
    # pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    
    if subsample > 0:
        print('---subsampling pts---')
        pts *= scale_factor
        pts, col = sample_pointcloud(pts, col, target_distance=subsample)
    else:
        pts *= scale_factor
        
    save_point_cloud_to_ply(pts, f'{outdir}', col*255)

def get_reconstructed_scene(
    outdir,
    model,
    device,
    image_size,
    filelist,
    schedule,
    niter,
    min_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    scenegraph_type,
    winsize,
    refid,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(
        imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True
    )
    output = inference(pairs, model, device, batch_size=batch_size)

    mode = (
        GlobalAlignerMode.PointCloudOptimizer
        if len(imgs) > 2
        else GlobalAlignerMode.PairViewer
    )
    scene = global_aligner(output, device=device, mode=mode)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )

    outfile = get_3D_model_from_scene(
        outdir,
        scene,
        min_conf_thr,
        as_pointcloud,
        mask_sky,
        clean_depth,
        transparent_cams,
        cam_size,
    )

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap("jet")
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs



def save_camera_poses(poses: torch.tensor, file_names: list, path: str) -> None:
    # convert pose to nparray
    M_ext = dict()
    assert len(file_names) == len(poses)
    for pose, view_name in zip(poses, file_names):
        image_file = f"{view_name}"
        M_ext[view_name] = pose.tolist()
    file_path = os.path.join("./estimation", path)
    if not os.path.isfile(file_path):
        os.mknod(file_path)
    with open(file_path, "w") as f:
        format_M_ext = {"extrinsics": M_ext}
        json.dump(format_M_ext, f)


def save_point_to_ply(points: np.ndarray, file_name: str) -> None:
    _, dim = points.shape
    if dim == 4:
        points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(os.path.join("./estimation/", file_name), pcd)


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def read_dtu_poses(path, scan, img_ids):
    cam_file = "{0}/{1}/cameras.npz".format(path, f'scan{scan}')
    if not os.path.exists(cam_file):
        print(f'scan{scan} but using 114 cameras.npz')
        cam_file = os.path.join(
            path, "scan114", "cameras.npz"
        )
    
    n_images = 49
    camera_dict = np.load(cam_file)
    scale_mats = [
        camera_dict["scale_mat_%d" % idx].astype(np.float32)
        for idx in range(n_images)
    ]
    world_mats = [
        camera_dict["world_mat_%d" % idx].astype(np.float32)
        for idx in range(n_images)
    ]

    # scale focal length according to dust3r
    scale_h, scale_w = (
        384 * 1.0 / 1200,   #576
        512 * 1.0 / 1600,   #768
    )

    files_path = []
    intrinsics_all = []
    pose_all = []
    for id_ in img_ids:
        # K, pose
        scale_mat, world_mat = scale_mats[id_], world_mats[id_]
        P = world_mat @ scale_mat
        P = P[:3, :4]

        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        intrinsics_all.append((intrinsics))
        pose_all.append((pose))

        files_path.append(f"../data/{args.dataset}/scan{scan}/image/{id_:06d}.png")

    return intrinsics_all, pose_all, files_path

def read_json_and_get_transforms(json_path, scan):
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    

    cx, cy = data['cx'], data['cy']
    fx, fy = data['fl_x'], data['fl_y']
    w, h = data['w'], data['h']

    # Extract transformation matrices
    intrinsic_all = []
    c2ws = []
    files_path = []

    # set scale based on what dust3r uses
    if scan == 'garden':
        # garden
        scale_h, scale_w = (
            320 * 1.0 / h,
            512 * 1.0 / w,
        )
    else:
        # stump, kitchen, bonsai, bicycle
        scale_h, scale_w = (
            336 * 1.0 / h,
            512 * 1.0 / w,
        )

    intrinsic = np.eye(3)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy

    intrinsic[0, :] *= scale_w
    intrinsic[1, :] *= scale_h
    
    for i, frame in enumerate(data['frames']):
        name = (frame['file_path'].split('/')[-1])
        if scan == 'garden' and name in ['DSC08116.JPG', 'DSC08121.JPG', 'DSC08140.JPG']:
            files_path.append(os.path.join(f'../data/{args.dataset}/{scan}/image', name))
            c2w = np.array(frame['transform_matrix'])
            c2ws.append(c2w)
            intrinsic_all.append(intrinsic)
        elif scan == 'stump' and name in ['_DSC9307.JPG', '_DSC9313.JPG', '_DSC9328.JPG']:
            files_path.append(os.path.join(f'../data/{args.dataset}/{scan}/image', name))
            c2w = np.array(frame['transform_matrix'])
            c2ws.append(c2w)
            intrinsic_all.append(intrinsic)

        if len(files_path) >= 3:
            break

    # Concatenate along the first dimension
    transforms_array = np.stack(c2ws, axis=0)
    intrinsic_array = np.stack(intrinsic_all, axis=0)
    
    return intrinsic_array, transforms_array, files_path

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
    if args.dataset == 'dtu':
        scan_id = int(args.scan_id)
        assert scan_id in [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]
        path = os.path.join(f'../data/{args.dataset}')
        intrinsic_all, c2w_all, files_path = read_dtu_poses(path, scan_id, [22, 25, 28])
    elif args.dataset == 'mipnerf':
        scan_id = str(args.scan_id)
        assert scan_id in ['garden', 'stump']
        path = os.path.join(f'../data/{args.dataset}/{scan_id}/{scan_id}.json')
        intrinsic_all, c2w_all, files_path = read_json_and_get_transforms(path, scan_id)
    else:
        raise KeyError('args.dataset supports only dtu or mipnerf')


    # use the first two images to reconstruct the scene
    images = load_images(files_path, size=512)       # [384, 512]
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, fx_and_fy=True #PointCloudOptimizer
    )

    
    # scale poses to 0.225 (what dust3r predicts - just for stability)
    E = np.stack(c2w_all)
    scale_factor = 0.225 / np.mean(np.linalg.norm(E[:, :3, 3]))
    E[:, :3, 3] *= scale_factor
    scene.preset_pose([E[i] for i in range(args.views)], [True for _ in range(args.views)])
    scene.preset_focal([K.diagonal()[:2].mean() for K in intrinsic_all], [True for _ in range(args.views)])

    loss = scene.compute_global_alignment(
        init="mst", niter=niter, schedule=schedule, lr=lr
    )

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()                                                                                                                                                                                                                                                                                         
    poses = scene.get_im_poses()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    if args.dataset == 'dtu':
        save_points_path = f'../data/{args.dataset}/scan{scan_id}/{str(scan_id)}.ply'
    else:
        save_points_path = f'../data/{args.dataset}/{scan_id}/{str(scan_id)}.ply'
    get_3D_pc_from_scene(save_points_path, scene, min_conf_thr=10, as_pointcloud=True, mask_sky=True, 
                            clean_depth=True, scale_factor=1/scale_factor, subsample=0.025)