import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from PIL import Image
from glob import glob
import spurfies.utils.general as utils
from spurfies.utils import rend_util
from spurfies.feat_utils import (
    load_cam,
    load_pair,
    scale_camera,
    FeatExt,
)


def get_trains_ids(data_dir, scan, num_views=0, for_interp=False):
    # if num_views <= 0:
    #     raise NotImplementedError
    if data_dir == "dtu" and num_views == 49:
        return list(range(49))
    if data_dir == "dtu":
        train_ids_all = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        return train_ids_all[:num_views]
    else:
        raise NotImplementedError


def get_eval_ids(data_dir, scan_id=None):
    # from regnerf/pixelnerf
    train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
    exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
    test_idx = [i for i in range(49) if i not in train_idx + exclude_idx]
    return test_idx


class DTUDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        img_res,
        scan_id=0,
        num_views=-1,
        data_dir_root=None,
        mode="train",
    ):

        self.data_dir, self.scan_id, self.num_views = data_dir, scan_id, num_views
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        assert num_views in [3, -1]
        self.mode, self.plot_id = mode, 0
        self.sampling_idx = None
        self.use_pixel_centers = False
        self.scan_id = scan_id
        instance_dir = os.path.join(data_dir_root, data_dir, "scan{0}".format(scan_id))
        image_dir = "{0}/image".format(instance_dir)
        self.gt_depths_dir = "{0}/depth".format(instance_dir)

        self.cam_file = "{0}/cameras.npz".format(instance_dir)
        if not os.path.exists(self.cam_file) and int(scan_id) < 200:
            self.cam_file = os.path.join(
                data_dir_root, data_dir, "scan114", "cameras.npz"
            )
        print(image_dir)
        
        assert os.path.exists(image_dir), "Data directory is empty"
        assert os.path.exists(self.cam_file), "Data directory is empty"
        if data_dir == "dtu":
            image_paths = sorted(utils.glob_imgs(image_dir))[:49]
        else:
            image_paths = sorted(utils.glob_imgs(image_dir))

        self.n_images = len(image_paths)
        camera_dict = np.load(self.cam_file)
        scale_mats = [
            camera_dict["scale_mat_%d" % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]
        world_mats = [
            camera_dict["world_mat_%d" % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]

        # low res
        rgb = rend_util.load_rgb(image_paths[0])  # (C, H, W)
        scale_h, scale_w = (
            self.img_res[0] * 1.0 / rgb.shape[0],
            self.img_res[1] * 1.0 / rgb.shape[1],
        )

        # mask
        mask_path = os.path.join(data_dir_root, data_dir, "eval_mask")
        if data_dir == "dtu":
            maskf_fn = lambda x: os.path.join(
                mask_path, f"scan{scan_id}", "mask", f"{x:03d}.png"
            )
        else:
            raise NotImplementedError

        self.rgb_images = []
        self.masks = []
        self.intrinsics_all = []
        self.pose_all = []
        self.scale_factor = scale_mats[0][0, 0]

        for id_, path in enumerate(image_paths):
            # K, pose
            scale_mat, world_mat = scale_mats[id_], world_mats[id_]
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            # pose - scale
            intrinsics[0, :] *= scale_w
            intrinsics[1, :] *= scale_h
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            # mask
            if (
                data_dir == "dtu"
            ):
                fname = maskf_fn(id_)
                with open(fname, "rb") as imgin:
                    mask_image = np.array(Image.open(imgin), dtype=np.float32)
                    if len(mask_image.shape) < 3:
                        mask_image = np.repeat(mask_image[:, :, None], 3, axis=2)
                    mask_image = mask_image[..., :3] / 255.0
                    mask_image = (mask_image == 1).astype(np.float32)
                if scale_h != 1 or scale_w != 1:
                    mask_image = cv2.resize(
                        mask_image,
                        (self.img_res[1], self.img_res[0]),
                        cv2.INTER_NEAREST,
                    )
                    mask_image = (mask_image > 0.5).astype(np.float32)
                    _mask = mask_image
                mask_image = mask_image.transpose(2, 0, 1)  # (3, H, W)
                mask_image = mask_image.reshape(3, -1).transpose(
                    1, 0
                )  # -> (3, H*W) -> (H*W, 3)
                self.masks.append(torch.from_numpy(mask_image).float())

            # rgb
            img = rend_util.load_rgb(path)  # (H, W, 3)
            # rgb - scale
            if scale_h != 1 or scale_w != 1:
                img = cv2.resize(
                    img,
                    (self.img_res[1], self.img_res[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            rgb = img.transpose(2, 0, 1)  # (3, H, W)
            rgb = rgb.reshape(3, -1).transpose(1, 0)  # -> (3, H*W) -> (H*W, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        # VisMVS feat from NeuSurf
        self.pair = load_pair(
            f"{data_dir_root}/dtu/DTU_pixelnerf/dtu_scan24/cam4feat/pair.txt"
        )
        self.num_src = 2
        self.depth_cams = torch.stack(
            [
                torch.from_numpy(
                    load_cam(
                        f'{data_dir_root}/dtu/DTU_pixelnerf/dtu_scan24/cam4feat/cam_{self.pair["id_list"][i].zfill(8)}_flow3.txt',
                        256,
                        1,
                    )
                ).to(torch.float32)
                for i in range(3)
            ],
            dim=0,
        )
        self.feat_img_scale = 2
        self.cams_hd = torch.stack(  # upsample of 2 from depth_cams, not 1200 * 1600
            [
                scale_camera(self.depth_cams[i], self.feat_img_scale) for i in range(3)
            ]  # NOTE: hard code
        )

        # self.images -> [22, 25, 28] training images
        self.images_lis = sorted(
            glob(
                os.path.join(
                    f"{data_dir_root}/dtu/DTU_pixelnerf/dtu_scan{scan_id}",
                    "image/*.png",
                )
            )
        )
        self.images_np = (
            np.stack([cv2.imread(im_name) for im_name in self.images_lis]) / 256.0
        )
        self.images = torch.from_numpy(
            self.images_np.astype(np.float32)
        ).cpu()  # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]

        self.img_res_f = self.images.shape[-3:-1]
        # [n_images, 3, 768, 1024]
        self.rgb_2xd = torch.stack(
            [
                F.interpolate(
                    self.images[idx]
                    .reshape(-1, 3)
                    .permute(1, 0)
                    .view(1, 3, *self.img_res_f),  # 1200 x 1600
                    size=(384 * self.feat_img_scale, 512 * self.feat_img_scale),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                for idx in range(3)
            ],
            dim=0,
        )  # v3hw
        mean = torch.tensor([0.485, 0.456, 0.406]).float().cpu()
        std = torch.tensor([0.229, 0.224, 0.225]).float().cpu()
        self.rgb_2xd = (self.rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(
            1, 3, 1, 1
        )
        self.size = torch.from_numpy(scale_mats[0]).float()[0, 0] * 2
        self.center = torch.from_numpy(scale_mats[0]).float()[:3, 3]

        feat_ext = FeatExt().cuda()
        feat_ext.eval()
        for p in feat_ext.parameters():
            p.requires_grad = False
        feats = []

        for start_i in range(0, 3):
            eval_batch = self.rgb_2xd[start_i : start_i + 1]
            feat2 = feat_ext(eval_batch.cuda())[2]  # .detach().cpu()
            feats.append(feat2)
        self.feats = torch.cat(feats, dim=0)  # [V, 32, 384, 512]
        self.feats.requires_grad = False

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        select an image with random N rays/pixels
        """
        # if self.num_views >= 1:
        train_ids = self.trains_ids()
        unobs_ids = [x for x in range(self.n_images) if x not in train_ids]
        if self.mode == "train":
            idx = train_ids[random.randint(0, self.num_views - 1)]
        elif self.mode == "plot":
            eval_ids = get_eval_ids(data_dir=self.data_dir, scan_id=self.scan_id)
            if len(eval_ids) == 0:
                eval_ids = unobs_ids
            idx = eval_ids[self.plot_id]
            self.plot_id = (self.plot_id + 1) % len(eval_ids)

        uv = np.mgrid[0 : self.img_res[0], 0 : self.img_res[1]].astype(np.int32)
        uv_patch = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv_patch.reshape(2, -1).transpose(1, 0)

        if self.use_pixel_centers:
            uv += torch.rand_like(uv)  # 0.5

        # VisMVS
        local = None
        if (
            idx in train_ids
            and self.num_views != -1
            and self.num_views == 3
            and self.mode != "plot"
        ):
            id = self.get_id_enum(self.data_dir, idx)
            src_id = self.get_id_enum(
                self.data_dir, self.get_src_id(self.data_dir, idx)
            )

            local = {}
            local["depth_cams"] = self.depth_cams[[id]]
            local["size"] = self.size
            local["center"] = self.center
            local["feat"] = self.feats[id]
            local["feat_src"] = self.feats[src_id]
            local["cam"] = self.cams_hd[id]
            local["src_cams"] = self.cams_hd[src_id]
            local["H"] = self.H
            local["W"] = self.W
            local["src_idxs"] = src_id

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx],
            "local_data": local,
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "mask": self.masks[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]
            ground_truth["mask"] = self.masks[idx][self.sampling_idx, :]

        return idx, sample, ground_truth

    def get_src_id(self, data_dir, ref_id):
        # TODO: add for more views
        if data_dir == "dtu" and self.num_views == 3:
            dict_src = {
                22: [25, 28],
                25: [22, 28],
                28: [22, 25],
            }
        return dict_src[ref_id]

    def get_id_enum(self, data_dir, ref_id):
        if data_dir == "dtu" and self.num_views == 3:
            dict_src = {
                22: 0,
                25: 1,
                28: 2,
            }
            if isinstance(ref_id, list):
                return [dict_src[rid] for rid in ref_id]
            else:
                return dict_src[ref_id]

    def trains_ids(self):
        return get_trains_ids(
            data_dir=self.data_dir, scan=f"scan{self.scan_id}", num_views=self.num_views
        )

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    if (
                        k == "local_data"
                    ):
                        ret[k] = entry[0][k]
                    else:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)["scale_mat_0"]
