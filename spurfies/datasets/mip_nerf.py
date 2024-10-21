import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from glob import glob
import spurfies.utils.general as utils
from spurfies.utils import rend_util
from spurfies.feat_utils import (
    FeatExt,
)
import json


def get_trains_ids(data_dir, scan, num_views=0, for_interp=False):
    return [0, 1, 2]

def get_eval_ids(data_dir, scan_id=None, mode='train'):
    if mode == 'train':
        return [0, 1, 2]
    else:
        return np.arange(7).tolist()

class MipDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        img_res,
        scan_id=0,
        num_views=-1,
        data_dir_root=None,
        mode='train',
    ):

        if scan_id == 'garden':
            img_res = [420, 648]    # garden
        elif scan_id == 'stump':
            img_res = [413, 622]    # stump
        else:
            raise NotImplementedError

        self.data_dir, self.scan_id, self.num_views = data_dir, scan_id, num_views
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        self.mode, self.plot_id = mode, 0
        self.sampling_idx = None
        self.use_pixel_centers = False
        self.scan_id = scan_id
        self.mode = mode

        instance_dir = os.path.join(data_dir_root, data_dir)

        if self.mode == 'train':
            image_dir = f"{instance_dir}/{self.scan_id}/image"
        elif self.mode == 'eval':
            image_dir = f"{instance_dir}/{self.scan_id}/image_eval"
        else:
            raise NotImplementedError

        self.cam_file = f"{instance_dir}/{self.scan_id}/{self.scan_id}.json"
        print(image_dir, self.cam_file)

        assert os.path.exists(image_dir), "Data directory is empty"
        assert os.path.exists(self.cam_file), "Data directory is empty"

        image_paths = sorted(utils.glob_imgs(image_dir))

        self.n_images = len(image_paths)

        with open(self.cam_file, "r") as f:
            data = json.load(f)

        cx, cy = data["cx"], data["cy"]
        fx, fy = data["fl_x"], data["fl_y"]
        w, h = data["w"], data["h"]

        scale_h, scale_w = (
            self.img_res[0] * 1.0 / h,
            self.img_res[1] * 1.0 / w,
        )

        intrinsic = np.eye(3)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy

        # -----------------
        # intrnsic vismvsnet
        intrinsic_vismvsnet = np.eye(4)
        intrinsic_vismvsnet[0, 0] = fx
        intrinsic_vismvsnet[1, 1] = fy
        intrinsic_vismvsnet[0, 2] = cx
        intrinsic_vismvsnet[1, 2] = cy
        img_res_vismvsnet = [768, 1024]  # [720, 1280]
        scale_h_vismvsnet, scale_w_vismvsnet = (
            img_res_vismvsnet[0] * 1.0 / h,
            img_res_vismvsnet[1] * 1.0 / w,
        )
        intrinsic_vismvsnet[0, :] *= scale_w_vismvsnet
        intrinsic_vismvsnet[1, :] *= scale_h_vismvsnet

        # -----------------

        intrinsic[0, :] *= scale_w
        intrinsic[1, :] *= scale_h

        K, c2ws, scale_mats = [], [], []
        for i, frame in enumerate(data["frames"]):
            # make sure camera corresponds to the order image is read
            name = frame["file_path"].split("/")[-1]
            if self.scan_id == 'garden':
                if name in ["DSC08116.JPG", "DSC08121.JPG", "DSC08140.JPG"]:  # garden
            # if name in ["DSC08115.JPG",
            #             "DSC08116.JPG",
            #             "DSC08117.JPG",
            #             "DSC08118.JPG",
            #             "DSC08119.JPG",
            #             "DSC08120.JPG",
            #             "DSC08121.JPG",
            #             "DSC08122.JPG",
            #             "DSC08123.JPG",
            #             "DSC08124.JPG",
            #             "DSC08125.JPG",
            #             "DSC08140.JPG"]: # garden eval
                    c2w = np.array(frame["transform_matrix"])
                    c2ws.append(c2w)
                    scale_mats.append(np.eye(4))
                    K.append(intrinsic)

            elif self.scan_id == 'stump':
                if name in ['_DSC9307.JPG', '_DSC9313.JPG', '_DSC9328.JPG']:    # stump
            # stump eval
            # if name in [
            #     '_DSC9217.JPG',
            #     '_DSC9235.JPG',
            #     '_DSC9309.JPG',
            #     '_DSC9311.JPG',
            #     '_DSC9326.JPG',
            #     '_DSC9329.JPG',
            #     '_DSC9338.JPG',
            # ]:
                    c2w = np.array(frame["transform_matrix"])
                    c2ws.append(c2w)

                    scale_mats.append(np.eye(4))
                    K.append(intrinsic)

        self.rgb_images = []
        self.gray_images = []
        self.rgb_smooth = []
        self.masks = []
        self.intrinsics_all = []
        self.pose_all = []
        self.scale_factor = scale_mats[0][0, 0]

        for id_, path in enumerate(image_paths):
            # K, c2w (pose)
            intrinsic, scale_mat, pose = (
                K[id_],
                scale_mats[id_],
                c2ws[id_],
            )

            # print(intrinsic)
            # print(pose)
            # print("-" * 50)

            self.intrinsics_all.append(torch.from_numpy(intrinsic).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            img = rend_util.load_rgb(path)  # (H, W, 3)
            # print('img: ', img.shape)

            mask_image = np.ones_like(img)
            mask_image = mask_image.transpose(2, 0, 1)  # (3, H, W)
            mask_image = mask_image.reshape(3, -1).transpose(
                1, 0
            )  # -> (3, H*W) -> (H*W, 3)
            self.masks.append(torch.from_numpy(mask_image).float())

            rgb = img.transpose(2, 0, 1)  # (3, H, W)
            rgb = rgb.reshape(3, -1).transpose(1, 0)  # -> (3, H*W) -> (H*W, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())


    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        """
        select an image with random N rays/pixels
        """
        if self.num_views >= 1:
            train_ids = self.trains_ids()
            unobs_ids = train_ids
            if self.mode == "train":
                idx = train_ids[random.randint(0, self.num_views - 1)]
            elif self.mode == "plot":
                eval_ids = train_ids
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


    def trains_ids(self):
        return [0, 1, 2]

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                ret = {}
                for k in entry[0].keys():
                    if (
                        k == "local_data"
                    ):  # and self.mode != 'plot':
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
        return np.eye(4)
