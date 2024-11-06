import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from glob import glob
import spurfies.utils.general as utils
from spurfies.utils import rend_util
import json


def get_trains_ids(data_dir, scan, num_views=0, for_interp=False):
    return [0, 1, 2]

def get_eval_ids(data_dir, scan_id=None, mode='train'):
    return [0, 1, 2]

class OwnData(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        img_res,
        scan_id=0,
        num_views=-1,
        data_dir_root=None,
        mode='train',
    ):

        self.data_dir, self.scan_id, self.num_views = data_dir, scan_id, num_views
        
        self.mode, self.plot_id = mode, 0
        self.sampling_idx = None
        self.use_pixel_centers = False
        self.scan_id = scan_id
        self.mode = mode

        instance_dir = os.path.join(data_dir_root, data_dir)
        
        image_dir = f"{instance_dir}/{self.scan_id}/image"
        mask_dir = f"{instance_dir}/{self.scan_id}/mask"
        
        self.cam_file = f"{instance_dir}/{self.scan_id}/{self.scan_id}.json"
        print(image_dir, self.cam_file)

        assert os.path.exists(image_dir), "Data directory is empty"
        assert os.path.exists(self.cam_file), "Data directory is empty"

        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_paths = sorted(utils.glob_imgs(mask_dir))
        
        self.n_images = len(image_paths)

        with open(self.cam_file, "r") as f:
            data = json.load(f)

        cx, cy = data["cx"], data["cy"]
        fx, fy = data["fl_x"], data["fl_y"]
        w, h = data["w"], data["h"]
        img_res = [h,w] # [height first, width second]
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        
        intrinsic = np.eye(3)
        intrinsic[0, 0] = fx
        intrinsic[1, 1] = fy
        intrinsic[0, 2] = cx
        intrinsic[1, 2] = cy

        K, c2ws, scale_mats = [], [], []
        for i, frame in enumerate(data["frames"]):
            # make sure camera corresponds to the order image is read
            c2w = np.array(frame["transform_matrix"])
            c2ws.append(c2w)
            scale_mats.append(np.eye(4))
            K.append(intrinsic)

        self.rgb_images = []
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

            self.intrinsics_all.append(torch.from_numpy(intrinsic).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            img = rend_util.load_rgb(path)  # (H, W, 3)
            
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
            uv += 0.5

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
