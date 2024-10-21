import numpy as np
import os
from subprocess import check_output
import cv2
import trimesh
import json


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=0.4)  # , axis_radius=0.1)
    box = trimesh.primitives.Box(extents=(2.2, 2.2, 2.2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    w2cs, Ks = [], []
    for pose in poses:

        # a camera is visualized with 8 line segments.
        # pose = np.linalg.inv(pose)
        pos = pose[:3, 3]  # c2w

        w2cs.append(pose)

        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        # dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir  # * 3

        segs = np.array(
            [
                [pos, a],
                [pos, b],
                [pos, c],
                [pos, d],
                [a, b],
                [b, c],
                [c, d],
                [d, a],
                [pos, o],
            ]
        )
        segs = trimesh.load_path(segs)
        objects.append(segs)
    return objects, w2cs, Ks

name = 'kitchen'
pose_path = f'/BS/local_diffusion/work/s-volsdf/data_s_volsdf/Mip_NeRF_360_{name}/transforms.json'

# Read the JSON file
with open(pose_path, "r") as f:
    data = json.load(f)

vis_pose = []
for i, frame in enumerate(data['frames']):
    # if frame['file_path'].split('/')[-1] in ['DSC08056.JPG', 'DSC08085.JPG', 'DSC08115.JPG']:
    #     print(i)
    
    c2w = np.array(frame['transform_matrix']) # c2w
    vis_pose.append(c2w)

# """
E = np.stack(vis_pose)
scene_length = np.mean(np.linalg.norm(E[:, :3, 3]))
print('scene length: ', scene_length)


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
    
    return pointcloud_normalized, camera_positions_normalized, centroid, scale_factor

pts = trimesh.load(
    f"/BS/local_diffusion/work/s-volsdf/data_s_volsdf/Mip_NeRF_360_{name}/{name}_dust3r_ngp.ply"
)
# print(pts.bounds)
pts.vertices, trans, centroid, scale_factor = normalize_pointcloud_and_cameras(pts.vertices, E[:, :3, 3])
# print(pts.bounds)
print(pts.bounds[0], pts.bounds[1])

pts.export(f'/BS/local_diffusion/work/s-volsdf/data_s_volsdf/Mip_NeRF_360_{name}/{name}_dust3r_ngp_scaled_2.ply')

E[:, :3, 3] = trans
vis_pose = [E[i] for i in range(E.shape[0])]

# Update data['transforms'] with vis_pose
for i, pose in enumerate(vis_pose):
    if i < len(data['frames']):
        data['frames'][i]['transform_matrix'] = pose.tolist()    # c2w

# Save the updated JSON
with open(f'/BS/local_diffusion/work/s-volsdf/data_s_volsdf/Mip_NeRF_360_{name}/transform_scale_2.json', 'w') as f:
    json.dump(data, f, indent=4)


# %%

poses_vis, _, Ks = visualize_poses(vis_pose)
scene = trimesh.Scene()
scene.add_geometry(poses_vis)
scene.add_geometry(pts)
scene.show()

"""


