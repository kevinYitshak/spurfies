from loguru import logger
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from spurfies.utils import rend_util
from spurfies.model.embedder import *
from spurfies.model.density import LaplaceDensity
from spurfies.model.ray_sampler import ErrorBoundSampler_pn, UniformSampler, PNeRFSampler
from torch_knnquery import VoxelGrid
from spurfies.model.utils import (
    load_neural_points,
    query,
    get_keypoint_data,
    mask_to_batch_ray_idx,
    plot,
    compute_tv_norm,
    tv_regul,
)
import torch as th
import spurfies.feat_utils as feat_utils
import json
import os

class PointVolSDF(nn.Module):
    def __init__(self, conf, scan_id, dataset):
        super().__init__()
        self.conf = conf
        self.scan_id = scan_id
        self.dataset = dataset

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

        self.conf.rbf = 45
        print("conf.rbf: ", self.conf.rbf)
 
        if self.scan_id in ['garden', 'stump'] and self.dataset == 'mipnerf':
            self._voxel_grid_neural = VoxelGrid(
                (0.025, 0.025, 0.025),
                (3, 3, 3),
                (3, 3, 3),
                26,  # 8 neighbors
                20000,
                (-2, -2, -2, 2,  2,  2)
            )
        else:
            self._voxel_grid_neural = VoxelGrid(
                (0.025, 0.025, 0.025),
                (3, 3, 3),
                (3, 3, 3),
                26,  # 8 neighbors
                20000,
                (-1, -1, -1, 1, 1, 1),
            )

        logger.info("---- VOXEL GRID ----")
        logger.info("-" * 30)

        self._init_neural_info()
        logger.info("loaded neural_pts onto voxelgrid")

        self.position_encoding, pos_in_dim = get_embedder(
            multires=6, input_dims=3
        )
        self.view_encoding, dir_in_dim = get_embedder(
            multires=3, input_dims=3
        )
        self.F_color = nn.Sequential(
            nn.Linear(conf.feature_vector_size + pos_in_dim, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
        )

        self.F_geometry = nn.Sequential(
            nn.Linear(conf.feature_vector_size // 2 + 3, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
        )

        self.T = nn.Sequential(nn.Linear(256, 1, bias=True))
    
        self.R = nn.Sequential(
            nn.Linear(256+dir_in_dim, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 3, bias=True),
            nn.Sigmoid(),
        )


        self.density = LaplaceDensity(**conf.get_config("density"))

        self.ray_sampler = ErrorBoundSampler_pn(
            self.scene_bounding_sphere, **conf.get_config("ray_sampler")
        )

    @staticmethod
    def _init_neural_feats(neural_feats):
        scale = 1e-4  # 1e-1 used
        neural_feats.uniform_(-scale, scale)

    @staticmethod
    def init_latent_codes(neural_feats):
        # Initialize the tensor with a normal distribution
        torch.nn.init.normal_(neural_feats, mean=0.0, std=0.01)
        # Apply max_norm constraint
        with torch.no_grad():
            norms = neural_feats.norm(dim=-1, keepdim=True)
            desired = torch.clamp(norms, max=1)
            neural_feats *= desired / (norms + 1e-7)

    def _init_neural_info(self):
        
        # use fps neural pts
        if self.dataset == 'dtu':
            self.conf.pointcloud_path = f"./data/{self.dataset}/scan{self.scan_id}/{self.scan_id}.ply"
        elif self.dataset == 'mipnerf':
            self.conf.pointcloud_path = f"./data/{self.dataset}/{self.scan_id}/{self.scan_id}.ply"
        else:
            raise NotImplementedError

        if self.conf.pointcloud_path is not None:
            neural_data = load_neural_points(
                self.conf.pointcloud_path, vox_res=self.conf.vox_res
            ) 
            neural_pts, neural_colors = (
                neural_data["pts"].float(),
                neural_data["colors"].float(),
            )

            # buffer for neural pts
            self.register_buffer(
                "neural_pts",
                torch.empty((len(neural_pts), 3), dtype=torch.float32, device="cuda"),
            )

            # params for neural feats
            self.register_parameter(
                "neural_feats_color",
                nn.Parameter(
                    torch.empty(
                        ((len(neural_pts)), self.conf.feature_vector_size),
                        dtype=torch.float32,
                        device="cuda",
                    )
                ),
            )

            self.register_parameter(
                "neural_feats_geometry",
                nn.Parameter(
                    torch.empty(
                        ((len(neural_pts)), self.conf.feature_vector_size // 2),
                        dtype=torch.float32,
                        device="cuda",
                    )
                ),
            )

            self.neural_pts.copy_(neural_pts.to(device=self.neural_pts.device))
            self.weights_buffer = torch.zeros(
                (len(self.neural_pts)), device=self.neural_pts.device
            )

            # init color feat
            self._init_neural_feats(self.neural_feats_color.data)
            # init geo feat
            self.init_latent_codes(self.neural_feats_geometry.data)

        if self.conf.pointcloud_path is not None:
            # from tetra-nerf
            if self.conf.initialize_colors:
                assert neural_colors.shape == (len(self.neural_pts), 3)
                colors = (
                    neural_colors.float().to(self.neural_feats_color.device)
                    * 2.0
                    / 255.0
                    - 1.0
                )
                self.neural_feats_color.data[:, :3] = colors
            print("+-+" * 10)
            print(f"pointcloud initialized from file {self.conf.pointcloud_path}:")
            print(f"num neural points: {len(self.neural_pts)}")
            print("+-+" * 10)
        else:
            raise RuntimeError("The pointcloud_path must be specified.")

    def filter_points(self, points, cam_loc, ray_dirs):
        _cam_loc = cam_loc[self.ray_mask]
        _ray_dirs = ray_dirs[self.ray_mask]
        sqp = (
            torch.zeros((*self.valid_neural_pts_mask.shape, 3), dtype=torch.float32)
            .float()
            .to(points.device)
        )
        sqp[self.valid_neural_pts_mask] = points.clone().detach()

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
        dist = torch.clamp(torch.norm(x_pi, dim=-1), min=1e-12).clone().detach()
        self.intrp_weights = torch.exp(
            -((dist * self.conf.rbf) ** 2)
        )
        self.norm = torch.zeros(self.num_valid_pts, device=x_pi.device)
        self.norm.index_add_(0, self.idx, self.intrp_weights)

    def get_sdf_eval(self, inputs):
        inputs = inputs.unsqueeze(1)  # [10000, 1, 3]

        self._voxel_grid_neural.set_pointset(
            self.neural_pts.unsqueeze(0),
            torch.full(
                (1,),
                fill_value=len(self.neural_pts),
                device="cuda",
                dtype=torch.int,
            ),
        )

        neighbor_idx, shading_pts, mask, ray_mask = query(
            self._voxel_grid_neural,
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
                neighbor_idx,
                valid_mask,
                self.neural_pts,
                self.neural_feats_color,
                self.neural_feats_geometry,
            )
            shading_pos = shading_pts[self.idx, :]  # [num_valid_pairs, 3]
            x_pi = shading_pos - kp["pos"]
            self.compute_weights(x_pi)
            # -------------------------
            agg_sdf = self.get_sdf(x_pi, kp)
            sdf_filler[ray_mask] = agg_sdf.squeeze(-1)

        return sdf_filler

    def get_sdf(self, x_pi, kp: dict, x=None, curvature=False):
        # make sure concat order is correct
        field_in = torch.concat([kp["feat_geometry"], x_pi], dim=-1)
        # aggregator
        feat_geomtry = self.F_geometry(field_in)

        sdf = self.T(feat_geomtry)

        weighted_sdf = self.intrp_weights.unsqueeze(-1) * sdf
        intrp_sdf = torch.zeros(self.num_valid_pts, 1, device=x_pi.device)
        intrp_sdf.index_add_(0, self.idx, weighted_sdf)
        agg_sdf = intrp_sdf / self.norm.unsqueeze(-1)

        return agg_sdf

    def get_gradients(self, sdf, points):
        gradients = torch.autograd.grad(
            sdf,
            points,
            torch.ones_like(sdf),
            retain_graph=True,
            create_graph=True,
        )[0]
        return gradients

    def get_color(self, x_pi, kp, ray_dirs):

        _ray_dirs = ray_dirs[self.ray_mask]  # [valid_rays, 3]

        pos_relative_vectors = self.position_encoding(x_pi)
        field_in_color = torch.concat([pos_relative_vectors, kp["feat"]], dim=-1)
        feat_color = self.F_color(field_in_color)

        weighted_feat = self.intrp_weights.unsqueeze(-1) * feat_color
        intrp_feat = torch.zeros(self.num_valid_pts, 256, device=ray_dirs.device)
        intrp_feat.index_add_(0, self.idx, weighted_feat)
        agg_feat_color = intrp_feat / self.norm.unsqueeze(-1)

        _ray_dirs = _ray_dirs.unsqueeze(1).expand(-1, self.conf.max_shading_pts, -1)[
            self.valid_neural_pts_mask
        ]
        encoded_dir = self.view_encoding(_ray_dirs)

        mlp_out = [encoded_dir, agg_feat_color]
        colors = self.R(torch.concat(mlp_out, dim=-1))

        return colors

    def sdf_importance(self, inputs):

        # print(inputs.shape)
        inputs = inputs.unsqueeze(1)  # [10000, 1, 3]

        self._voxel_grid_neural.set_pointset(
            self.neural_pts.unsqueeze(0),
            torch.full(
                (1,),
                fill_value=len(self.neural_pts),
                device="cuda",
                dtype=torch.int,
            ),
        )

        neighbor_idx_neural, shading_pts_neural, mask_neural, ray_mask_neural = query(
            self._voxel_grid_neural,
            inputs.contiguous(),
            self.conf.k,
            self.conf.r,
            1,
        )

        sdf_filler = torch.ones((inputs.shape[0]), device=inputs.device) * 1000
        if shading_pts_neural.shape[0] > 0:
            valid_mask_neural = (
                neighbor_idx_neural >= 0
            )  # contains bool of neighbor idx for a shading pt
            neighbor_idx_neural[~valid_mask_neural] = (
                0  # for valid indices during index select
            )

            num_valid_pts_neural = neighbor_idx_neural.shape[0]
            idx_neural = mask_to_batch_ray_idx(valid_mask_neural)  # [num_valid_pairs]
            shading_pts_neural = inputs[mask_neural]
            # shading_pts_gcl.requires_grad_(True)

            # with torch.enable_grad():
            kp_neural = get_keypoint_data(
                neighbor_idx_neural,
                valid_mask_neural,
                self.neural_pts,
                self.neural_feats_color,
                self.neural_feats_geometry,
            )
            shading_pos_neural = shading_pts_neural[
                idx_neural, :
            ]  # [num_valid_pairs, 3]
            x_pi_neural = shading_pos_neural - kp_neural["pos"]

            dist_neural = (
                torch.clamp(torch.norm(x_pi_neural, dim=-1), min=1e-12).clone().detach()
            )

            intrp_weights_neural = torch.exp(-((dist_neural * self.conf.rbf) ** 2))
            norm_neural = torch.zeros(num_valid_pts_neural, device=x_pi_neural.device)
            norm_neural.index_add_(0, idx_neural, intrp_weights_neural)

            field_in_neural = torch.concat(
                [kp_neural["feat_geometry"], x_pi_neural], dim=-1
            )
            # aggregator
            feat_geomtry_neural = self.F_geometry(field_in_neural)
            sdf_neural = self.T(feat_geomtry_neural)

            weighted_sdf_neural = intrp_weights_neural.unsqueeze(-1) * sdf_neural
            intrp_sdf_neural = torch.zeros(
                num_valid_pts_neural, 1, device=x_pi_neural.device
            )
            intrp_sdf_neural.index_add_(0, idx_neural, weighted_sdf_neural)
            agg_sdf_neural = intrp_sdf_neural / norm_neural.unsqueeze(-1)

            sdf_filler[ray_mask_neural] = agg_sdf_neural.squeeze(-1)
        return sdf_filler

    def pseudo_sdf(self, inputs):

        inputs = inputs.unsqueeze(1)  # [N, 1, 3]

        self._voxel_grid_neural.set_pointset(
            self.neural_pts.unsqueeze(0),
            torch.full(
                (1,),
                fill_value=len(self.neural_pts),
                device="cuda",
                dtype=torch.int,
            ),
        )

        neighbor_idx_pseudo, shading_pts_pseudo, mask_pseudo, ray_mask_pseudo = query(
            self._voxel_grid_neural,
            inputs.contiguous(),
            self.conf.k,
            self.conf.r,
            1,
        )

        agg_sdf_pseudo = torch.ones((inputs.shape[0]), device=inputs.device) * 1000
        if shading_pts_pseudo.shape[0] > 0:
            valid_mask_pseudo = (
                neighbor_idx_pseudo >= 0
            )  # contains bool of neighbor idx for a shading pt
            neighbor_idx_pseudo[~valid_mask_pseudo] = (
                0  # for valid indices during index select
            )

            num_valid_pts_pseudo = neighbor_idx_pseudo.shape[0]
            idx_pseudo = mask_to_batch_ray_idx(valid_mask_pseudo)  # [num_valid_pairs]
            shading_pts_pseudo = inputs[mask_pseudo]
            # shading_pts_gcl.requires_grad_(True)

            # with torch.enable_grad():
            kp_pseudo = get_keypoint_data(
                neighbor_idx_pseudo,
                valid_mask_pseudo,
                self.neural_pts,
                self.neural_feats_color,
                self.neural_feats_geometry,
            )
            shading_pos_pseudo = shading_pts_pseudo[
                idx_pseudo, :
            ]  # [num_valid_pairs, 3]
            x_pi_pseudo = shading_pos_pseudo - kp_pseudo["pos"]
            # -------------------------
            dist_pseudo = (
                torch.clamp(torch.norm(x_pi_pseudo, dim=-1), min=1e-12).clone().detach()
            )
            
            intrp_weights_pseudo = torch.exp(-((dist_pseudo * self.conf.rbf) ** 2))
            norm_pseudo = torch.zeros(num_valid_pts_pseudo, device=x_pi_pseudo.device)
            norm_pseudo.index_add_(0, idx_pseudo, intrp_weights_pseudo)

            # TODO: swap concat to test disent
            field_in_neural = torch.concat(
                [kp_pseudo["feat_geometry"], x_pi_pseudo], dim=-1
            )
            # aggregator
            feat_geomtry_pseudo = self.F_geometry(field_in_neural)
            sdf_pseudo = self.T(feat_geomtry_pseudo)

            weighted_sdf_pseudo = intrp_weights_pseudo.unsqueeze(-1) * sdf_pseudo
            intrp_sdf_pseudo = torch.zeros(
                num_valid_pts_pseudo, 1, device=x_pi_pseudo.device
            )
            intrp_sdf_pseudo.index_add_(0, idx_pseudo, weighted_sdf_pseudo)
            agg_sdf_pseudo = intrp_sdf_pseudo / norm_pseudo.unsqueeze(-1)

        return agg_sdf_pseudo

    def get_rays(self, cam_loc, ray_dirs):
        ray_dirs = ray_dirs.reshape(-1, 3)
        cam_loc = cam_loc.unsqueeze(1).repeat(1, ray_dirs.shape[0], 1).reshape(-1, 3)

        z_vals = self.ray_sampler.get_z_vals(
            ray_dirs, cam_loc, self
        )  # (N_rays, N_samples)
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        return points, z_vals, cam_loc, ray_dirs

    def get_importance_rays(self, cam_loc, ray_dirs, model, fast=-1, iter_step=None):
        ray_dirs = ray_dirs.reshape(-1, 3)
        cam_loc = cam_loc.unsqueeze(1).repeat(1, ray_dirs.shape[0], 1).reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(
            ray_dirs, cam_loc, model, fast, iter_step
        )  # [500, 98]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        return points, z_vals, cam_loc, ray_dirs

    def sdf_reg(self):

        inputs = self.neural_pts.unsqueeze(1)  # [10000, 1, 3]

        self._voxel_grid_neural.set_pointset(
            self.neural_pts.unsqueeze(0),
            torch.full(
                (1,),
                fill_value=len(self.neural_pts),
                device="cuda",
                dtype=torch.int,
            ),
        )

        neighbor_idx_reg, shading_pts_reg, mask_reg, ray_mask_reg = query(
            self._voxel_grid_neural,
            inputs.contiguous(),
            self.conf.k,
            self.conf.r,
            1,
        )

        if shading_pts_reg.shape[0] > 0:
            valid_mask_reg = (
                neighbor_idx_reg >= 0
            )  # contains bool of neighbor idx for a shading pt
            neighbor_idx_reg[~valid_mask_reg] = (
                0  # for valid indices during index select
            )

            num_valid_pts_reg = neighbor_idx_reg.shape[0]
            idx_reg = mask_to_batch_ray_idx(valid_mask_reg)  # [num_valid_pairs]
            shading_pts_reg = inputs[mask_reg]
            # shading_pts_reg.requires_grad_(True)

            # with torch.enable_grad():
            kp_reg = get_keypoint_data(
                neighbor_idx_reg,
                valid_mask_reg,
                self.neural_pts,
                self.neural_feats_color,
                self.neural_feats_geometry,
            )
            shading_pos_reg = shading_pts_reg[idx_reg, :]  # [num_valid_pairs, 3]
            x_pi_reg = shading_pos_reg - kp_reg["pos"]
            # -------------------------
            dist_reg = (
                torch.clamp(torch.norm(x_pi_reg, dim=-1), min=1e-12).clone().detach()
            )
            
            intrp_weights_reg = torch.exp(-((dist_reg * self.conf.rbf) ** 2))
            norm_reg = torch.zeros(num_valid_pts_reg, device=x_pi_reg.device)
            norm_reg.index_add_(0, idx_reg, intrp_weights_reg)

            # TODO: swap concat to test disent
            field_in_reg = torch.concat([kp_reg["feat_geometry"], x_pi_reg], dim=-1)
            # aggregator
            feat_geomtry_reg = self.F_geometry(field_in_reg)
            sdf_reg = self.T(feat_geomtry_reg)

            weighted_sdf_reg = intrp_weights_reg.unsqueeze(-1) * sdf_reg
            intrp_sdf_reg = torch.zeros(num_valid_pts_reg, 1, device=x_pi_reg.device)
            intrp_sdf_reg.index_add_(0, idx_reg, weighted_sdf_reg)
            agg_sdf_reg = intrp_sdf_reg / norm_reg.unsqueeze(-1)

        return agg_sdf_reg

    # interpolate SDF zero-crossing points
    def find_surface_points(self, sdf, d_all, device="cuda"):
        sdf[sdf == 1000] = torch.nan
        # shape of sdf and d_all: only inside
        sdf_bool_1 = sdf[..., 1:] * sdf[..., :-1] < 0
        # only find backward facing surface points, not forward facing
        sdf_bool_2 = sdf[..., 1:] < sdf[..., :-1]
        sdf_bool = torch.logical_and(sdf_bool_1, sdf_bool_2)

        max, max_indices = torch.max(sdf_bool, dim=2)
        network_mask = max > 0
        d_surface = torch.zeros_like(network_mask, device=device).float()

        sdf_0 = torch.gather(
            sdf[network_mask], 1, max_indices[network_mask][..., None]
        ).squeeze()
        sdf_1 = torch.gather(
            sdf[network_mask], 1, max_indices[network_mask][..., None] + 1
        ).squeeze()
        d_0 = torch.gather(
            d_all[network_mask], 1, max_indices[network_mask][..., None]
        ).squeeze()
        d_1 = torch.gather(
            d_all[network_mask], 1, max_indices[network_mask][..., None] + 1
        ).squeeze()
        d_surface[network_mask] = (sdf_0 * d_1 - sdf_1 * d_0) / (sdf_0 - sdf_1)

        return d_surface, network_mask

    def forward(self, input, fast=-1):
        
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        iter_step = input.get("iter_step", 1)
        local_loss = torch.tensor(0.0).cuda().float()
        pseudo_pts_loss = torch.tensor(0.0).cuda().float()

        local_data = input["local_data"]
        # load neural pts to voxel here
        self._voxel_grid_neural.set_pointset(
            self.neural_pts.unsqueeze(0),
            torch.full(
                (1,),
                fill_value=len(self.neural_pts),
                device="cuda",
                dtype=torch.int,
            ),
        )

        # ==== ray dirs and loc ====
        # [500, 512, 3]
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # we should use unnormalized ray direction for depth (sphere vs pinhole)
        ray_dirs_tmp, _ = rend_util.get_camera_params(
            uv, torch.eye(4).to(pose.device)[None], intrinsics
        )
        depth_scale = ray_dirs_tmp[0, :, 2:]

        points, _, cam_loc, ray_dirs = self.get_importance_rays(
            cam_loc, ray_dirs, self, fast, iter_step
        )
        # [N_rays, N_samples, :]

        # query neighbors
        # shading_pts # [rays, samples, 3]
        self.neighbor_idx, shading_pts, self.mask, self.ray_mask = query(
            self._voxel_grid_neural,
            points,
            self.conf.k,
            self.conf.r,
            self.conf.max_shading_pts,
        )

        self.valid_neural_pts_mask = self.mask[self.ray_mask]  # [rays, samples]
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

            kp = get_keypoint_data(
                self.neighbor_idx,
                valid_mask,
                self.neural_pts,
                kp_feat=self.neural_feats_color,
                kp_geometry=self.neural_feats_geometry,
            )

            shading_pos = shading_pts[self.idx, :]  # [num_valid_pairs, 3]
            x_pi = shading_pos - kp["pos"]

            # compute intrp weights and make it self. var
            self.compute_weights(x_pi)

            agg_sdf = self.get_sdf(x_pi, kp, x=shading_pts)
            gradients = self.get_gradients(agg_sdf, shading_pts)
            colors = self.get_color(
                x_pi,
                kp,
                ray_dirs
            )

            sdf_filler = (
                torch.ones_like(deltas, dtype=torch.float32).float().to(points.device)
            ) * 1000
            sdf_filler[self.valid_neural_pts_mask] = agg_sdf

            gradients_filler = (
                torch.zeros((*deltas.shape[:2], 3), dtype=torch.float32)
                .float()
                .to(points.device)
            )
            gradients_filler[self.valid_neural_pts_mask] = gradients

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

            # TODO: local loss
            # zero crossing neusurf
            if local_data is not None and self.training:
                size, center = local_data["size"].unsqueeze(0), local_data[
                    "center"
                ].unsqueeze(0)
                size = size[:1].to("cuda")
                center = center[:1].to("cuda")

                cam = local_data["cam"]  # 2, 4, 4
                src_cams = local_data["src_cams"]  # m, 2, 4, 4
                feat_src = local_data["feat_src"]

                d_surface, network_mask = self.find_surface_points(
                    sdf_filler.squeeze(-1).unsqueeze(0),
                    z_values.squeeze(-1).unsqueeze(0),
                )
                d_surface = d_surface.squeeze(0)
                network_mask = network_mask.squeeze(0)
                object_mask = network_mask
                point_surface = (
                    cam_loc[self.ray_mask]
                    + ray_dirs[self.ray_mask] * d_surface[:, None]
                )
                point_surface_wmask = point_surface[network_mask & object_mask].to(
                    "cuda"
                )
                local_loss = feat_utils.get_local_loss(
                    point_surface_wmask,
                    None,
                    local_data["feat"].unsqueeze(0),
                    cam.unsqueeze(0),
                    feat_src.unsqueeze(0),
                    src_cams.unsqueeze(0),
                    size,
                    center,
                    network_mask.reshape(-1),
                    object_mask.reshape(-1),
                )

            dist_map = torch.sum(
                weights_values
                / (weights_values.sum(-1, keepdim=True) + 1e-10)
                * z_values.squeeze(-1),
                -1,
            )
            pts_rendered = (
                cam_loc[self.ray_mask] + ray_dirs[self.ray_mask] * dist_map[:, None]
            )  # [N, 3]
            # compute sdf
            sdf_rendered_points = self.pseudo_sdf(pts_rendered)
            sdf_rendered_points_wmask = sdf_rendered_points  # [object_mask]
            sdf_rendered_points_0 = torch.zeros_like(sdf_rendered_points_wmask)
            pseudo_pts_loss = F.l1_loss(
                sdf_rendered_points_wmask, sdf_rendered_points_0, reduction="mean"
            )

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

        output["local_loss"] = local_loss
        output["pseudo_pts_loss"] = pseudo_pts_loss

        # tv_reg
        tv_loss = tv_regul(
            self._voxel_grid_neural,
            self.neural_pts,
            self.neural_feats_geometry,
            self.conf.k,
            self.conf.r,
        )
        output["tv_loss"] = tv_loss

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
