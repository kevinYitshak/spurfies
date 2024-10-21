import torch
from torch import nn
import spurfies.utils.general as utils
import math
import numpy as np
from helpers.help import logger
import torch.nn.functional as F


def anneal_linearly(t, val0, val1):
    if t >= 1:
        return val1
    elif t <= 0:
        return val0
    return val0 + (val1 - val0) * np.minimum(t, 1.0)


class VolSDFLoss(nn.Module):
    def __init__(
        self,
        rgb_loss,
        local_weight,
        pseudo_weight,
        eikonal_weight,
        rgb_weight=1.0,
        tv_weight=0.0,
    ):
        super().__init__()
        self.local_weight = local_weight
        self.pseudo_weight = pseudo_weight
        self.eikonal_weight = eikonal_weight
        self.rgb_weight = rgb_weight
        self.tv_weight = tv_weight

        logger.info(
            f"loss lambda: RGB_{rgb_weight} EK_{eikonal_weight}"
        )
        self.rgb_loss = utils.get_class(rgb_loss)(reduction="mean")

        self.iter_step = 0

    def get_rgb_loss(self, rgb_values, rgb_gt, model_outputs=None, t=0):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth["rgb"].cuda()
        mask_gt = ground_truth["mask"].cuda()

        output = {}
        output["rgb_loss"] = self.get_rgb_loss(model_outputs["rgb_values"], rgb_gt)
        if "grad_theta" in model_outputs:
            output["eikonal_loss"] = self.get_eikonal_loss(model_outputs["grad_theta"])
        else:
            output["eikonal_loss"] = torch.tensor(0.0).cuda().float()

        # tv
        if "tv_loss" in model_outputs and self.tv_weight > 0:
            output["tv_loss"] = model_outputs["tv_loss"]
        else:
            output["tv_loss"] = torch.tensor(0.0).cuda().float()

        # mask loss
        if "weights" in model_outputs:
            weights_sum = model_outputs["weights"].sum(-1, keepdim=True)
            output["mask_loss"] = F.binary_cross_entropy(
                weights_sum.clip(1e-3, 1.0 - 1e-3), mask_gt.squeeze()[:, 0][..., None]
            )
        else:
            output["mask_loss"] = torch.tensor(0.0).cuda().float()

        # local loss
        if "local_loss" in model_outputs:
            output["local_loss"] = model_outputs["local_loss"]
        else:
            output["local_loss"] = torch.tensor(0.0).cuda().float()

        # pseudo loss
        if "pseudo_pts_loss" in model_outputs and self.pseudo_weight > 0:
            output["pseudo_loss"] = model_outputs["pseudo_pts_loss"]
        else:
            output["pseudo_loss"] = torch.tensor(0.0).cuda().float()

        # total loss
        output["loss"] = (
            self.rgb_weight * output["rgb_loss"]
            + self.eikonal_weight * output["eikonal_loss"]
            + self.tv_weight * output["tv_loss"]
            + self.local_weight * output["local_loss"]
            + self.pseudo_weight * output["pseudo_loss"]
            + output["mask_loss"]
        )

        self.iter_step += 1

        return output
