from loguru import logger
import torch.nn as nn
import numpy as np

from spurfies.utils import rend_util
from spurfies.model.embedder import *
from spurfies.model.density import LaplaceDensity
from spurfies.model.ray_sampler import ErrorBoundSampler, UniformSampler, PNeRFSampler
from torch_knnquery import VoxelGrid
from spurfies.model.utils import (
    load_neural_points,
    query,
    get_keypoint_data,
    mask_to_batch_ray_idx,
    plot,
)


class PointVolSDF(nn.Module):
    def __init__(self, conf, scan_id):
        super().__init__()
        self.scan_id = scan_id
        self.conf = conf
        self.feature_vector_size = conf.get_int("feature_vector_size")
        self.scene_bounding_sphere = conf.get_float(
            "scene_bounding_sphere", default=1.0
        )
        self.white_bkgd = conf.get_bool("white_bkgd", default=False)
        self.bg_color = (
            torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0]))
            .float()
            .cuda()
        )

        self._voxel_grid = VoxelGrid(
            (0.025, 0.025, 0.025),
            (2, 2, 2),
            (3, 3, 3),
            26,
            1100000,
            (-1, -1, -1, 1, 1, 1),
        )
        logger.info("---- VOXEL GRID ----")
        logger.info("-" * 30)

        self._init_neural_info()
        logger.info("loaded neural_pts onto voxelgrid")

        self.position_encoding, mlp_in_dim = get_embedder(multires=4, input_dims=3)
        self.direction_encoding, dir_in_dim = get_embedder(multires=6, input_dims=3)
        self.F = nn.Sequential(
            nn.Linear(conf.feature_vector_size + mlp_in_dim, 256, bias=True),    # no pos_enc
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            # nn.LeakyReLU(inplace=True),
        )
        self.T = nn.Sequential(nn.Linear(256, 1, bias=True))
        self.R = nn.Sequential(
            nn.Linear(256 + dir_in_dim, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 3, bias=True),
            nn.Sigmoid(),
        )

        self.density = LaplaceDensity(**conf.get_config("density"))

        self.ray_sampler = UniformSampler(
            self.scene_bounding_sphere, **conf.get_config("ray_sampler")
        )

    @staticmethod
    def _init_neural_feats(neural_feats):
        scale = 1e-4  # 1e-1 used
        neural_feats.uniform_(-scale, scale)

    def _init_neural_info(self):
        self.conf.pointcloud_path = '/BS/local_diffusion/work/dust3r/estimation/blended_points_icp_scaled_down/1.ply'
        if self.conf.pointcloud_path is not None:
            neural_data = load_neural_points(self.conf.pointcloud_path)
            neural_pts = neural_data["pts"].float()

            # buffer for neural pts
            self.register_buffer(
                "neural_pts",
                torch.empty((len(neural_pts), 3), dtype=torch.float32, device="cuda"),
            )

            # params for neural feats
            self.register_parameter(
                "neural_feats",
                nn.Parameter(
                    torch.empty(
                        ((len(neural_pts)), self.conf.feature_vector_size),
                        dtype=torch.float32,
                        device="cuda",
                    )
                ),
            )

            self.neural_pts.copy_(neural_pts.to(device=self.neural_pts.device))
            self.weights_buffer = torch.zeros(
                (len(self.neural_pts)), device=self.neural_pts.device
            )

            self._init_neural_feats(self.neural_feats.data)
            # params for neural feats
            
        if self.conf.pointcloud_path is not None:
            neural_data = load_neural_points(self.conf.pointcloud_path)
            neural_pts = neural_data["pts"].float()
            self.neural_pts.copy_(neural_pts.to(device=self.neural_pts.device))

            self._init_neural_feats(self.neural_feats.data)

            if self.conf.initialize_colors:
                assert "colors" in neural_data
                assert neural_data["colors"].dtype == torch.uint8
                assert neural_data["colors"].shape == (len(self.neural_pts), 3)
                colors = (
                    neural_data["colors"].float().to(self.neural_feats.device)
                    * 2.0
                    / 255.0
                    - 1.0
                )
                self.neural_feats.data[:, :3] = colors
            print("+-+" * 10)
            print(f"pointcloud initialized from file {self.conf.pointcloud_path}:")
            print(f"num neural points: {len(self.neural_pts)}")
            print("+-+" * 10)
        else:
            raise RuntimeError("The pointcloud_path must be specified.")

    # TODO
    def grow_points(self, input):
        # shoot rays and compute opacity
        # modify self.neural_pts, and self.neural_feats
        pass

    def filter_points(self, points, cam_loc, ray_dirs):
        _cam_loc = cam_loc[self.ray_mask]
        _ray_dirs = ray_dirs[self.ray_mask]
        sqp = (
            torch.zeros((*self.valid_neural_pts_mask.shape, 3), dtype=torch.float32)
            .float()
            .to(points.device)
        )  # [500, 256, 3]
        # print(sqp.shape, valid_neural_pts_mask.shape, points.shape)
        sqp[self.valid_neural_pts_mask] = points.clone().detach()

        # get depths for selected_query_pts
        # cam_loc:  torch.Size([500, 3]) torch.Size([500, 1, 3])
        # ray_dirs:  torch.Size([500, 3]) torch.Size([500, 1, 3])
        # t:  torch.Size([500, 256, 1])
        t = ((sqp - _cam_loc.unsqueeze(1)) / _ray_dirs.unsqueeze(1)).nanmean(
            dim=-1, keepdim=True
        )
        t_filler = torch.zeros_like(t).float().to(t.device)
        t_filler[self.valid_neural_pts_mask] = t[
            self.valid_neural_pts_mask
        ]  # [valid_pts]
        z_vals = t_filler
        # append one extra to calculate deltas
        _z_vals = torch.cat(
            [z_vals, torch.zeros(z_vals.shape[0], 1, 1).to(z_vals.device)], dim=1
        )
        deltas = _z_vals[:, 1:] - _z_vals[:, :-1]
        # for non-contributing pts set deltas = 0
        deltas[~self.valid_neural_pts_mask] = 0
        deltas = deltas.clamp_(min=0)

        # cal points on filtered z_vals
        # [valid_rays, max_shading_pts, 3]
        points = (_cam_loc.unsqueeze(1) + z_vals * _ray_dirs.unsqueeze(1))[
            self.valid_neural_pts_mask
        ]
        return points, z_vals, deltas

    def compute_weights(self, x_pi):
        dist = torch.clamp(torch.norm(x_pi, dim=-1), min=1e-12)#.clone().detach()
        self.intrp_weights = 1.0 / dist   # mesh fk up?!?!
        # self.intrp_weights = torch.exp(
        #     -((dist * 40) ** 2)
        # )   # color fk up?!?!
        self.norm = torch.zeros(self.num_valid_pts, device=x_pi.device)
        self.norm.index_add_(0, self.idx, self.intrp_weights)

        # normalize weights
        # norm_weight_holder = torch.zeros_like(self.intrp_weights, device=dist.device)
        # print('norm and idx: ', len(self.norm), self.idx.max())
        norm_per_neighbor = self.norm[self.idx]
        # print('intrp_weights: ', self.intrp_weights.shape, self.norm_per_neighbor.shape)
        self.intrp_weights_normalized = self.intrp_weights / norm_per_neighbor
        

    def get_sdf_eval(self, inputs): 
        inputs = inputs.unsqueeze(1)  # [10000, 1, 3]

        self._voxel_grid.set_pointset(
            self.neural_pts.unsqueeze(0),
            torch.full(
                (1,),
                fill_value=len(self.neural_pts),
                device="cuda",
                dtype=torch.int,
            ),
        )

        neighbor_idx, shading_pts, mask, ray_mask = query(
            self._voxel_grid,
            inputs.contiguous(),
            self.conf.k,
            self.conf.r,
            1,
        )
        # print(neighbor_idx.shape, shading_pts.shape, mask.shape, ray_mask.shape)

        sdf_filler = torch.ones((inputs.shape[0]), device=inputs.device) * 1000
        if shading_pts.shape[0] > 0:
            valid_mask = (
                neighbor_idx >= 0
            )  # contains bool of neighbor idx for a shading pt
            neighbor_idx[~valid_mask] = 0  # for valid indices during index select

            self.num_valid_pts = neighbor_idx.shape[0]
            self.idx = mask_to_batch_ray_idx(valid_mask)  # [num_valid_pairs]
            shading_pts = inputs[mask]

            # shading_pts.requires_grad_(True)
            # with torch.enable_grad():
            kp = get_keypoint_data(
                neighbor_idx, valid_mask, self.neural_pts, self.neural_feats
            )
            shading_pos = shading_pts[self.idx, :]  # [num_valid_pairs, 3]
            x_pi = shading_pos - kp["pos"]
            self.compute_weights(x_pi)
            agg_sdf, _ = self.get_sdf(x_pi, kp)
            sdf_filler[ray_mask] = agg_sdf.squeeze(-1)

        return sdf_filler

    def get_sdf(self, x_pi, kp: dict):

        pos_relative_vectors = self.position_encoding(x_pi)
        field_in = torch.concat([pos_relative_vectors, kp["feat"]], dim=-1)
        # field_in = torch.concat([x_pi, kp["feat"]], dim=-1)
        # field_in = self.intrp_weights_normalized.unsqueeze(-1) * kp["feat"]
        # intrp_field_in = torch.zeros(self.num_valid_pts, kp["feat"].shape[-1], device=x_pi.device)
        # intrp_field_in.index_add_(0, self.idx, field_in)
        # aggregator
        feat = self.F(field_in)
        # feat = self.F(intrp_field_in)

        sdf = self.T(feat)
        weighted_sdf = self.intrp_weights.unsqueeze(-1) * sdf
        intrp_sdf = torch.zeros(self.num_valid_pts, 1, device=x_pi.device)
        intrp_sdf.index_add_(0, self.idx, weighted_sdf)
        agg_sdf = intrp_sdf / self.norm.unsqueeze(-1)
        # agg_sdf = sdf
        return agg_sdf, feat

    def get_gradients(self, sdf, points):
        gradients = torch.autograd.grad(
            sdf,
            points,
            torch.ones_like(sdf),
            retain_graph=True,
            create_graph=True,
        )[0]
        return gradients

    def get_color(
        self,
        feat,
        ray_dirs,
        normals=None,
    ):

        _ray_dirs = ray_dirs[self.ray_mask]  # [valid_rays, 3]

        weighted_feat = self.intrp_weights.unsqueeze(-1) * feat
        intrp_feat = torch.zeros(self.num_valid_pts, 256, device=ray_dirs.device)
        intrp_feat.index_add_(0, self.idx, weighted_feat)
        agg_feat = intrp_feat / self.norm.unsqueeze(-1)

        _ray_dirs = _ray_dirs.unsqueeze(1).expand(-1, self.conf.max_shading_pts, -1)[
            self.valid_neural_pts_mask
        ]
        encoded_dir = self.direction_encoding(_ray_dirs)

        mlp_out = [encoded_dir, agg_feat]
        colors = self.R(torch.concat(mlp_out, dim=-1))

        return colors

    def get_rays(self, cam_loc, ray_dirs):
        ray_dirs = ray_dirs.reshape(-1, 3)
        cam_loc = cam_loc.unsqueeze(1).repeat(1, ray_dirs.shape[0], 1).reshape(-1, 3)

        z_vals = self.ray_sampler.get_z_vals(
            ray_dirs, cam_loc, self
        )  # (N_rays, N_samples)
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        return points, z_vals, cam_loc, ray_dirs

    def forward(self, input, fast=-1):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        iter_step = input.get("iter_step", 1)

        # load neural pts to voxel here
        self._voxel_grid.set_pointset(
            self.neural_pts.unsqueeze(0),
            torch.full(
                (1,),
                fill_value=len(self.neural_pts),
                device="cuda",
                dtype=torch.int,
            ),
        )
        # sys.exit()

        # ==== ray dirs and loc ====
        # [500, 512, 3]
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # we should use unnormalized ray direction for depth (sphere vs pinhole)
        ray_dirs_tmp, _ = rend_util.get_camera_params(
            uv, torch.eye(4).to(pose.device)[None], intrinsics
        )
        depth_scale = ray_dirs_tmp[0, :, 2:]
        # logger.info(f"depth_scale: ", depth_scale.shape)

        points, _, cam_loc, ray_dirs = self.get_rays(cam_loc, ray_dirs)
        # [N_rays, N_samples, :]

        # query neighbors
        # shading_pts # [rays, samples, 3]
        self.neighbor_idx, shading_pts, self.mask, self.ray_mask = query(
            self._voxel_grid,
            points,
            self.conf.k,
            self.conf.r,
            self.conf.max_shading_pts,
        )
        # plot(shading_pts, 'pn_shading_pts_begin')

        self.valid_neural_pts_mask = self.mask[self.ray_mask]  # [rays, samples]
        # print("-" * 50)
        # print("len hit points: ", len(shading_pts))
        # print("-" * 50)
        points_filler = None
        if shading_pts.shape[0] > 0:
            # TODO: check deltas
            shading_pts, z_values, deltas = self.filter_points(
                shading_pts, cam_loc, ray_dirs
            )
            shading_pts.requires_grad_(True)

            valid_mask = (
                self.neighbor_idx >= 0
            )  # contains bool of neighbor idx for a shading pt
            self.neighbor_idx[~valid_mask] = 0  # for valid indices during index select

            self.num_valid_pts = self.neighbor_idx.shape[0]
            self.idx = mask_to_batch_ray_idx(valid_mask)  # [num_valid_pairs]

            # logger.info(f"cam_loc: {cam_loc.shape}")
            # logger.info(f"z_vals: {z_vals.shape}")
            # logger.info(f"ray_dirs: {ray_dirs.shape}")
            # logger.info(f"valid_neural_pts_mask: {valid_neural_pts_mask.shape}")

            # logger.info(f"shading_pts: {shading_pts.shape}")

            kp = get_keypoint_data(
                self.neighbor_idx,
                valid_mask,
                self.neural_pts,
                kp_feat=self.neural_feats,
            )

            shading_pos = shading_pts[self.idx, :]  # [num_valid_pairs, 3]
            x_pi = shading_pos - kp["pos"]
            # compute intrp weights and make it self. var
            self.compute_weights(x_pi)
            # plot(kp["pos"], 'neural_pts_pn') # accounts for neighbors
            # plot(shading_pts, 'query_pts_pn')
            # sys.exit()

            agg_sdf, feat = self.get_sdf(x_pi, kp)
            gradients = self.get_gradients(agg_sdf, shading_pts)
            colors = self.get_color(feat, ray_dirs, normals=gradients)

            # density filler
            density_filler = (
                torch.zeros_like(deltas, dtype=torch.float32).float().to(points.device)
            )
            # convert sdf to density
            density_filler[self.valid_neural_pts_mask] = self.density(agg_sdf)

            # cal weights for rendering
            weights_values = self.volume_rendering(
                deltas[..., 0], density_filler[..., 0]
            )  # (N_rays, N_samples)

            # filler alles
            color_filler = (
                torch.zeros((*deltas.shape[:2], 3), dtype=torch.float32)
                .float()
                .to(points.device)
            )
            color_filler[self.valid_neural_pts_mask] = colors
            # --------------------------------------
            rgb_values = torch.sum(weights_values.unsqueeze(-1) * color_filler, 1)
            depth_values = torch.sum(
                weights_values * z_values.squeeze(-1), 1, keepdims=True
            ) / (weights_values.sum(dim=1, keepdims=True) + 1e-8)
            # TODO: check this we should scale rendered distance to depth along z direction
            depth_values = depth_scale[self.ray_mask] * depth_values
            acc_values = torch.sum(weights_values, -1, keepdim=True)

            # --------------------------------------
            if not self.training:
                normals = (
                    torch.zeros((*deltas.shape[:2], 3), dtype=torch.float32)
                    .float()
                    .to(points.device)
                )
                gradients = gradients.detach()
                _normals = gradients / gradients.norm(2, -1, keepdim=True)
                normals[self.valid_neural_pts_mask] = _normals
                normal_values = torch.sum(weights_values.unsqueeze(-1) * normals, 1)

            points_filler = (
                torch.zeros((*deltas.shape[:2], 3), dtype=torch.float32)
                .float()
                .to(points.device)
            )
            points_filler[self.valid_neural_pts_mask] = shading_pts

        # ------------------------------------------------------------------
        # Expand rendered values back to the original shape
        device = self.ray_mask.device
        rgb = torch.zeros((self.ray_mask.shape[0], 3), device=device)
        normal = torch.zeros((self.ray_mask.shape[0], 3), device=device)
        accumulation = torch.zeros(
            (self.ray_mask.shape[0], 1), dtype=torch.float32, device=device
        )
        depth = torch.full(
            (self.ray_mask.shape[0], 1),
            1,
            dtype=torch.float32,
            device=device,
        )
        weights = torch.zeros(
            (self.ray_mask.shape[0], self.conf.max_shading_pts), device=device
        )

        depth_vals = (
            torch.ones(
                (self.ray_mask.shape[0], self.conf.max_shading_pts),
                dtype=torch.float32,
                device=device,
            )
            * self.conf.ray_sampler.far
        )
        points = torch.zeros(
            (self.ray_mask.shape[0], self.conf.max_shading_pts, 3), device=device
        )
        if points_filler is not None:
            points[self.ray_mask] = points_filler

        if shading_pts.shape[0] > 0:  # fill masked values
            rgb[self.ray_mask] = rgb_values
            accumulation[self.ray_mask] = acc_values
            depth[self.ray_mask] = depth_values
            weights[self.ray_mask] = weights_values
            depth_vals[self.ray_mask] = (
                z_values.squeeze(-1) * depth_scale[self.ray_mask]
            )

        # ------------------------------------------------------------------

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (
                1.0 - acc_map[..., None]
            ) * self.bg_color.unsqueeze(0)

        output = {
            "rgb_values": rgb,
            "depth_values": depth,
            "depth_vals": depth_vals,
            "weights": weights,
            "xyz": points,
        }

        if not self.training:
            if len(shading_pts) > 0:
                normal[self.ray_mask] = normal_values
            output["normal_map"] = normal

        if self.training:
            output["grad_theta"] = gradients

        return output

    def volume_rendering(self, deltas, density):
        # print(deltas.shape, density.shape)

        # LOG SPACE
        free_energy = deltas * density
        shifted_free_energy = torch.cat(
            [torch.zeros(deltas.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1
        )  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        weights = alpha * transmittance  # probability of the ray hits something here

        return weights
