import plyfile
import numpy as np
import torch
from torch_scatter import scatter_min, scatter_mean

def construct_vox_points_closest(
    xyz_val, vox_res, partition_xyz=None, space_min=None, space_max=None
):
    # xyz, N, 3
    xyz = xyz_val if partition_xyz is None else partition_xyz
    if space_min is None:
        xyz_min, xyz_max = torch.min(xyz, dim=-2)[0], torch.max(xyz, dim=-2)[0]
        space_edge = torch.max(xyz_max - xyz_min) * 1.05
        xyz_mid = (xyz_max + xyz_min) / 2
        space_min = xyz_mid - space_edge / 2
    else:
        space_edge = space_max - space_min
        mask = xyz_val - space_min[None, ...]
        mask *= space_max[None, ...] - xyz_val
        mask = torch.prod(mask, dim=-1) > 0
        xyz_val = xyz_val[mask, :]
    construct_vox_sz = space_edge / vox_res
    xyz_shift = xyz - space_min[None, ...]
    sparse_grid_idx, inv_idx = torch.unique(
        torch.floor(xyz_shift / construct_vox_sz[None, ...]).to(torch.int32),
        dim=0,
        return_inverse=True,
    )
    xyz_centroid = scatter_mean(xyz_val, inv_idx, dim=0)
    xyz_centroid_prop = xyz_centroid[inv_idx, :]
    xyz_residual = torch.norm(xyz_val - xyz_centroid_prop, dim=-1)
    # print("xyz_residual", xyz_residual.shape)

    _, min_idx = scatter_min(xyz_residual, inv_idx, dim=0)
    # print("min_idx", min_idx.shape)
    return xyz_centroid, sparse_grid_idx, min_idx

def voxelize(pointcloud, vox_res):
    points_xyz_all = (
        [pointcloud] if not isinstance(pointcloud, list) else pointcloud
    )
    points_xyz_holder = torch.zeros(
        [0, 3], dtype=points_xyz_all[0].dtype, device="cuda"
    )
    for i in range(len(points_xyz_all)):
        points_xyz = points_xyz_all[i]
        vox_res = vox_res // (1.5**i)
        _, sparse_grid_idx, sampled_pnt_idx = construct_vox_points_closest(
            points_xyz.cuda()
            if len(points_xyz) < 80000000
            else points_xyz[:: (len(points_xyz) // 80000000 + 1), ...].cuda(),
            vox_res,
        )
        points_xyz = points_xyz[sampled_pnt_idx, :]
        points_xyz_holder = torch.cat([points_xyz_holder, points_xyz], dim=0)
    points_xyz_all = points_xyz_holder
    return points_xyz_all, sampled_pnt_idx

def load_neural_points(path, vox_res=None):
    plydata = plyfile.PlyData.read(path)
    x = plydata["vertex"]["x"]
    y = plydata["vertex"]["y"]
    z = plydata["vertex"]["z"]

    pointcloud = np.stack([x, y, z], axis=-1)
    pointcloud = torch.from_numpy(pointcloud).to('cuda')

    if vox_res is not None:
        pointcloud, sample_pt_idx = voxelize(pointcloud, vox_res)

    if "red" in plydata["vertex"]:
        r = plydata["vertex"]["red"]
        g = plydata["vertex"]["green"]
        b = plydata["vertex"]["blue"]
        color = np.stack([r, g, b], axis=-1)
        color = torch.from_numpy(color).to('cuda')

        if vox_res is not None:
            color = color[sample_pt_idx, :]

        return {
            "pts": pointcloud,
            "colors": color
        }
    
    return {
            "pts": pointcloud,
        }

def query(voxel_grid, inputs, k, r, max_shading_pts):
    
    num_rays = inputs.shape[0]
    sample_idx, sample_loc, ray_mask = voxel_grid.query(
                inputs.unsqueeze(0), k, r, max_shading_pts
            )
    sample_idx = sample_idx.to(dtype=torch.int64)
    ray_mask = ray_mask.bool()
    valid_neighbor_mask = sample_idx >= 0
    valid_pts_mask = valid_neighbor_mask.any(
        dim=-1, keepdim=True
    )  # [num_valid_rays, max_shading_pts, 1]
    mask = torch.zeros(
        (1, num_rays, max_shading_pts, 1), dtype=torch.bool, device=inputs.device
    )
    mask.masked_scatter_(ray_mask[..., None, None], valid_pts_mask)

    neighbor_idx = torch.masked_select(sample_idx, valid_pts_mask).view(
        -1, k
    )  # [num_valid_pts, k]
    shading_pts = torch.masked_select(sample_loc, valid_pts_mask).view(
        -1, 3
    )  # [num_valid_pts, 3]
    return neighbor_idx, shading_pts, mask.view(num_rays, -1), ray_mask.squeeze(0)

def query_geo(voxel_grid, inputs, k, r):
    
    num_rays = inputs.shape[0]
    sample_idx, sample_loc, ray_mask = voxel_grid.query(
                inputs.unsqueeze(0), k, r, 1, #config.max_shading_pts
            )
    sample_idx = sample_idx.to(dtype=torch.int64)
    ray_mask = ray_mask.bool()
    valid_neighbor_mask = sample_idx >= 0
    valid_pts_mask = valid_neighbor_mask.any(
        dim=-1, keepdim=True
    )  # [num_valid_rays, max_shading_pts, 1]
    mask = torch.zeros(
        (1, num_rays, 1, 1), dtype=torch.bool, device=inputs.device
    )
    mask.masked_scatter_(ray_mask[..., None, None], valid_pts_mask)

    neighbor_idx = torch.masked_select(sample_idx, valid_pts_mask).view(
        -1, k
    )  # [num_valid_pts, k]
    shading_pts = torch.masked_select(sample_loc, valid_pts_mask).view(
        -1, 3
    )  # [num_valid_pts, 3]
    return neighbor_idx, shading_pts, mask.view(num_rays, -1), ray_mask.squeeze(0)

def get_keypoint_data(
    neighbor_idx,
    mask,
    kp_pos = None,
    kp_feat = None,
    kp_geometry = None):
    k = neighbor_idx.shape[-1]
    res = {}
    data = []
    if kp_pos is not None:
        data.append(kp_pos)
    if kp_feat is not None:
        data.append(kp_feat)
    if kp_geometry is not None:
        data.append(kp_geometry)

    data = torch.cat(data, dim=-1)
    val_dim = data.shape[-1]
    selected = torch.index_select(
        data.view(-1, val_dim), 0, neighbor_idx.view(-1)
    ).view(-1, k, val_dim)
    masked = torch.masked_select(selected, mask[..., None]).view(-1, val_dim)
    if kp_pos is not None:
        res["pos"] = masked[:, :3]
        masked = masked[:, 3:]
    if kp_feat is not None:
        res["feat"] = masked[:, :kp_feat.shape[-1]]
        masked = masked[:, kp_feat.shape[-1]:]
    if kp_geometry is not None:
        res["feat_geometry"] = masked
    return res

def mask_to_batch_ray_idx(valid_neighbor_mask):
    """
    Arguments:
        valid_neighbor_mask: [num_valid_pts, k]
    Returns:
        batch_ray_idx: [num_valid_pairs]
    """
    num_valid_pts = valid_neighbor_mask.shape[0]
    source = torch.arange(num_valid_pts, device=valid_neighbor_mask.device).view(
        -1, 1
    )
    return torch.masked_select(source, valid_neighbor_mask)

def plot(pts, name):
    from plyfile import PlyData, PlyElement
    import numpy as np

    print("before: ", pts.shape)
    pts = pts.view(-1, 3).detach().cpu().numpy()
    # print("áfter", pts.shape)
    # os.mkdir('')
    vertex = np.core.records.fromarrays(pts.transpose(), names="x, y, z")
    elements = [PlyElement.describe(vertex, "vertex")]
    plydata = PlyData(elements, text=True)
    # # Save the PLY file
    plydata.write(f"./{name}.ply")
    print(f"saved {name}.ply")


def compute_tv_norm(values, losstype='l1'):
  """Returns TV norm for input values.

  Note: The weighting / masking term was necessary to avoid degenerate
  solutions on GPU; only observed on individual DTU scenes.
  """
  v00 = values[:, :-1, :-1]
  v01 = values[:, :-1, 1:]
  v10 = values[:, 1:, :-1]

  if losstype == 'l2':
    loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
  elif losstype == 'l1':
    loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
  else:
    raise ValueError('Not supported losstype.')

  return loss


def tv_regul(voxel_grid, kp_pos, kp_feat, k, r):
    """
    Var has attributes
    kp_pos: [n_kp, 3]
    kp_feat: [n_kp, feat_dim]
    """
    kp_pos = kp_pos.detach()

    n_kp = kp_pos.shape[0]
    total_kp = n_kp
    device = kp_pos.device
    x = kp_pos.view(n_kp, 3)  # [n_kp, 3]
    # VoxelGrid looses keypoints sometimes such that num_valid_kp does not have to be equal to B*n_kp
    # [num_valid_kp, k], _, [B, 1, n_kp, 50, 1]
    neighbor_idx, _, valid_kp_mask, _ = query_geo(voxel_grid, x.unsqueeze(1), k, r)
    # valid_kp_mask = valid_kp_mask[..., 0, :].view(n_kp, 1)
    # pad neighbor_idx to [B*n_kp, k]
    padded_neighbor_idx = torch.full(
        (n_kp, neighbor_idx.shape[-1]),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )   # [n_kp, 8]
    padded_neighbor_idx[..., 0] = torch.arange(n_kp, device=device).unsqueeze(0)
    padded_neighbor_idx.masked_scatter_(valid_kp_mask, neighbor_idx)
    neighbor_idx = padded_neighbor_idx
    # Remove origin kps as neighbors, if there are any other neighbors
    origin_idx = torch.arange(n_kp, device=device)[:, None]
    identity_mask = neighbor_idx == origin_idx
    valid_neighbor_mask = neighbor_idx >= 0
    enough_neighbor_mask = valid_neighbor_mask.int().sum(dim=-1, keepdim=True) > 1
    delete_mask = torch.logical_and(identity_mask, enough_neighbor_mask)
    neighbor_idx[delete_mask] = -1
    neighbor_idx = neighbor_idx #.flatten(0, 1)
    # Continue with modified neighbor_idx
    valid_neighbor_mask = neighbor_idx >= 0
    neighbor_idx[~valid_neighbor_mask] = 0  # for valid indices during index select
    idx = mask_to_batch_ray_idx(
        valid_neighbor_mask
    )  # [num_valid_pairs]
    origin_pos = kp_pos[idx]  # [num_valid_pairs, 3]
    neighbors = get_keypoint_data(
        neighbor_idx, valid_neighbor_mask, kp_pos, kp_geometry=kp_feat
    )
    # Compute neighbor weights as normalized inverse Euclidean distances
    weights = 1 / (
        torch.linalg.norm(neighbors["pos"] - origin_pos, dim=-1) + 1.0e-5
    )  # [num_valid_pairs]
    norm = torch.zeros(total_kp, device=device)
    norm.index_add_(0, idx, weights)
    # Compute weighted total variation
    origin_feat = kp_feat[idx]  # [num_valid_pairs, feat_dim]
    feat_dist = torch.linalg.norm(
         neighbors["feat_geometry"] - origin_feat, ord=1, dim=-1
    )  # [num_valid_pairs] (1/0.0001)
    weighted = weights * feat_dist
    tv = torch.zeros(total_kp, device=device)
    tv.index_add_(0, idx, weighted)
    # if self.opt.loss.tv_regul.use_normalization:
    tv = tv / norm
    tv = tv.view(n_kp)
    return tv.mean()