import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import matplotlib.cm as cm

from spurfies.utils import rend_util

def stacked_plot(indices, plot_data, path, epoch, img_res, plot_nimgs, resolution, grid_boundary, level=0):
    cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

    # plot images (2+300+2+300, 2+400+2, 3) 0 ~ 255
    img = plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices, save=False)

    # plot normal maps (300, 400, 3) 0 ~ 255
    normal = plot_normal_maps(plot_data['normal_map'], path, epoch, plot_nimgs, img_res, indices, save=False)

    # plot depth maps (300, 400, 3) 0 ~ 255
    depth = plot_depth_maps(plot_data['depth_map'], plot_data['acc'], path, epoch, plot_nimgs, img_res, indices, save=False)

    stack = [img[2:2+img_res[0], 2:2+img_res[1]], depth, normal]
    stack = [torch.from_numpy(_).permute(2, 0, 1) for _ in stack]
    return stack

def plot(implicit_network, indices, plot_data, path, epoch, img_res, plot_nimgs, resolution, grid_boundary, level=0):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot images
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], path, epoch, plot_nimgs, img_res, indices)

        # plot depth maps
        plot_depth_maps(plot_data['depth_map'], plot_data['acc'], path, epoch, plot_nimgs, img_res, indices)

        # concat output images to single large image
        images = []
        for name in ["rendering", "depth", "normal"]:
            images.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0]))) 
        images = [images[0][2:2+img_res[0], :]] + images[1:]

        images = np.concatenate(images, axis=1)
        cv2.imwrite('{0}/merge_{1}_{2}.png'.format(path, epoch, indices[0]), images)

    del plot_data
    torch.cuda.empty_cache()

    surface_traces = get_surface_trace(path=path, 
                                        epoch=epoch,
                                        sdf=lambda x: implicit_network(x)[:, 0],
                                        resolution=resolution,
                                        grid_boundary=grid_boundary,
                                        level=level
                                        )
    del surface_traces
 

def get_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_high_res_mesh(sdf, resolution=100, grid_boundary=[-2.0, 2.0], level=0, take_components=True):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100, grid_boundary)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=level,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    if take_components:
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float32)
        mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport

def get_surface_by_grid(grid_params, sdf, resolution=100, level=0, higher_res=False, splitn=100000):
    grid_params = grid_params * [[1.5], [1.0]]

    # params = PLOT_DICT[scan_id]
    input_min = torch.tensor(grid_params[0]).float()
    input_max = torch.tensor(grid_params[1]).float()

    if higher_res:
        # get low res mesh to sample point cloud
        grid = get_grid(None, 100, input_min=input_min, input_max=input_max, eps=0.0)
        z = []
        points = grid['grid_points']

        for i, pnts in enumerate(torch.split(points, splitn, dim=0)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        mesh_low_res = trimesh.Trimesh(verts, faces, normals)
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float64)
        mesh_low_res = components[areas.argmax()]

        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float().cuda()

        # Center and align the recon pc
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
        if torch.det(vecs) < 0:
            vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                           (recon_pc - s_mean).unsqueeze(-1)).squeeze()

        grid_aligned = get_grid(helper.cpu(), resolution, eps=0.01)
    else:
        grid_aligned = get_grid(None, resolution, input_min=input_min, input_max=input_max, eps=0.0)

    grid_points = grid_aligned['grid_points']

    if higher_res:
        g = []
        for i, pnts in enumerate(torch.split(grid_points, splitn, dim=0)):
            g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                               pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, splitn, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        if higher_res:
            verts = torch.from_numpy(verts).cuda().float()
            verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                       verts.unsqueeze(-1)).squeeze()
            verts = (verts + grid_points[0]).cpu().numpy()
        else:
            verts = verts + np.array([grid_aligned['xyz'][0][0], grid_aligned['xyz'][1][0], grid_aligned['xyz'][2][0]])

        meshexport = trimesh.Trimesh(verts, faces, normals)

        # CUTTING MESH ACCORDING TO THE BOUNDING BOX
        if higher_res:
            bb = grid_params
            transformation = np.eye(4)
            transformation[:3, 3] = (bb[1,:] + bb[0,:])/2.
            bounding_box = trimesh.creation.box(extents=bb[1,:] - bb[0,:], transform=transformation)

            meshexport = meshexport.slice_plane(bounding_box.facets_origin, -bounding_box.facets_normal)

    return meshexport

def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution, input_min=None, input_max=None, eps=0.1):
    if input_min is None or input_max is None:
        input_min = torch.min(points, dim=0)[0].squeeze().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def plot_depth_maps(depth_maps, acc, path, epoch, plot_nrow, img_res, indices, save=True):
    depth_maps_plot = lin2img(depth_maps[..., None], img_res)
    acc_plot = lin2img(acc[..., None], img_res)

    assert sum(depth_maps_plot.shape[:2]) + sum(acc_plot.shape[:2]) == 4
    tensor = visualize_depth(depth_maps_plot[0,0].cpu().numpy(), acc_plot[0,0].cpu().numpy())

    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    if not save:
        return tensor
    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}_{2}.png'.format(path, epoch, indices[0]))

def plot_normal_maps(normal_maps, path, epoch, plot_nrow, img_res, indices, save=True):
    normal_maps_plot = lin2img(normal_maps, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    if not save:
        return tensor
    img = Image.fromarray(tensor)
    img.save('{0}/normal_{1}_{2}.png'.format(path, epoch, indices[0]))

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res, indices, save=True):
    ground_true = ground_true.cpu()

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    if not save:
        return tensor
    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}_{2}.png'.format(path, epoch, indices[0]))


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

def visualize_depth(x, acc, lo=None, hi=None):
    """Visualizes depth maps."""

    depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    return visualize_cmap(
        x, acc, cm.get_cmap('turbo'), curve_fn=depth_curve_fn, lo=lo, hi=hi)

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :])
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]

def visualize_cmap(value,
                   weight,
                   colormap,
                   lo=None,
                   hi=None,
                   percentile=99.,
                   curve_fn=lambda x: x,
                   modulus=None,
                   matte_background=True):
    """Visualize a 2D image and a 2D weighting according to some colormap.
    Args:
        value: A 2D image.
        weight: A weight map, in [0, 1].
        colormap: A colormap function.
        lo: The lower bound to use when rendering, if None then use a percentile.
        hi: The upper bound to use when rendering, if None then use a percentile.
        percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
        curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
        modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
        matte_background: If True, matte the image over a checkerboard.
    Returns:
        A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(
        value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = np.finfo(np.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(
            np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    if colormap:
        colorized = colormap(value)[:, :, :3] # 2D
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return matte(colorized, weight) if matte_background else colorized