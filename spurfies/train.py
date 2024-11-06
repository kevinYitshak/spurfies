from helpers.help import logger
import os
from datetime import datetime
from pyhocon import ConfigFactory
import copy
import itertools
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from torch.nn.functional import grid_sample
from torch.utils.tensorboard import SummaryWriter

import spurfies.utils.general as utils
import spurfies.utils.plots as plt
from spurfies.utils import rend_util
from spurfies.datasets.dtu import DTUDataset    # DTU
from spurfies.datasets.mip_nerf import MipDataset
from spurfies.datasets.own_data import OwnData
from shutil import copyfile

class VolOpt:
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        # get configs
        self.hparams = copy.deepcopy(kwargs["args"])
        resolved = OmegaConf.to_container(
            self.hparams["vol"], resolve=True, throw_on_missing=True
        )
        self.conf = ConfigFactory.from_dict(resolved)
        self.batch_size = kwargs["batch_size"]
        self.exps_folder_name = self.hparams.exps_folder
        
        self.conf.get_string("dataset.data_dir")
        dataset_conf = self.conf.get_config("dataset")

        root = "./"
        self.expname = self.conf.get_string("train.expname")
        if dataset_conf.get_string('data_dir') == 'dtu':
            kwargs_scan_id = int(kwargs["scan"][4:])
            scan_id = (
                kwargs_scan_id
                if kwargs_scan_id != -1
                else self.conf.get_int("dataset.scan_id", default=-1)
            )
            self.scan_id = scan_id
        else:
            kwargs_scan_id = kwargs["scan"]
            scan_id = kwargs_scan_id
            self.scan_id = scan_id

        if scan_id != -1:
            self.expname = self.expname + "_{0}".format(scan_id)

        if kwargs["is_continue"] and kwargs["timestamp"] == "latest":
            if os.path.exists(
                os.path.join(root, self.hparams.exps_folder, self.expname)
            ):
                timestamps = os.listdir(
                    os.path.join(root, self.hparams.exps_folder, self.expname)
                )
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs["timestamp"]
            is_continue = kwargs["is_continue"]

        # create exps dirs
        utils.mkdir_ifnotexists(os.path.join(root, self.exps_folder_name))
        self.expdir = os.path.join(root, self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        self.plots_dir = os.path.join(self.expdir, self.timestamp, "plots")
        utils.mkdir_ifnotexists(self.plots_dir)
        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, "checkpoints")
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        utils.mkdir_ifnotexists(
            os.path.join(self.checkpoints_path, self.model_params_subdir)
        )
        utils.mkdir_ifnotexists(
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir)
        )

        # save configs
        with open(os.path.join(self.expdir, self.timestamp, "run.yaml"), "w") as f:
            OmegaConf.save(self.hparams, f)

        # dataset config
        logger.info("Loading data ...")
        if kwargs_scan_id != -1:
            dataset_conf["scan_id"] = kwargs_scan_id
        dataset_conf["data_dir_root"] = self.hparams.data_dir_root
        logger.info(f"    full resolution in VolOpt {dataset_conf['img_res']}")

        assert [self.hparams.max_h, self.hparams.max_w] == dataset_conf["img_res"]

        # generate dataset
        self.data_confs = [copy.deepcopy(dataset_conf) for _ in range(3)]
        self.gen_dataset(stg=2)  # full resolution
        self.gen_plot_dataset()
        self.stg = 2
        self.ds_len = len(self.train_dataset)  # number of training images

        # model
        conf_model = self.conf.get_config("model")
        self.model = utils.get_class(self.conf.get_string("train.model_class"))(
            conf=conf_model, scan_id=self.scan_id, dataset=dataset_conf.get_string('data_dir')
        )
        print(self.model)

        # load prior weights here
        if "disent" in self.conf.get_config("train").model_class:
            prior = torch.load("ckpt/local_prior.pt")["model_state_dict"]
            prior.pop("sdf_features")

            filtered_params = dict()
            cnt = torch.arange(0, 10, 2).repeat_interleave(2)
            for i, (k, v) in enumerate(prior.items()):
                if "local_sdf_field" in k:
                    name = f"F_geometry.{cnt[i]}." + ".".join(k.split(".")[4:])
                    filtered_params.update({f"{name}": v})
                if "density_branch.weight" in k:
                    name = f"T.0.weight"
                    filtered_params.update({f"{name}": v})
                if "density_branch.bias" in k:
                    name = f"T.0.bias"
                    filtered_params.update({f"{name}": v})
            self.model.load_state_dict(filtered_params, strict=False)
            logger.info("-" * 30)
            logger.info("LOADED PRIOR WEIGHTS")
            logger.info("-" * 30)

            print('neural_pts: ', self.model.neural_pts.shape)
            print('neural_feats_color: ', self.model.neural_feats_color.shape)
            print('neural_feats_geometry: ', self.model.neural_feats_geometry.shape)
            trianable_params = []

            sdf_feat = []
            for name, param in self.model.named_parameters():
                if "F_geometry" in name or "T.0" in name:
                    print("Non-trainable params: ", name)
                    param.requires_grad_(False)
                else:
                    print("trainable params: ", name)
                    trianable_params.append(param)

        if torch.cuda.is_available():
            self.model.cuda()

        # loss
        self.loss = utils.get_class(self.conf.get_string("train.loss_class"))(
            **self.conf.get_config("loss")
        )

        # optimizer
        self.lr = self.conf.get_float("train.learning_rate")

        if "disent" in self.conf.get_config("train").model_class:
            sdf_feat = [
                torch.Tensor(param) if isinstance(param, tuple) else param
                for param in sdf_feat
            ]
            self.optimizer = torch.optim.Adam(
                [
                    {
                        "params": sdf_feat,
                        "lr": 1e-2,
                    },
                    {"params": trianable_params, "lr": self.lr},
                ]
            )
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100_000, eta_min=3e-4, last_epoch=-1, verbose=False
        )

        # load ckpt
        self.start_epoch = 0
        self.iter_step, self.total_step = 0, 0
        ckpt_dir = self.conf.get_string("train.ckpt_dir", "")
        if is_continue:
            self.load_from_dir(
                dir=os.path.join(self.expdir, timestamp),
                checkpoint=kwargs["checkpoint"],
            )
        elif ckpt_dir != "":
            self.load_from_dir(dir=ckpt_dir, checkpoint="latest")

        # some parameters
        self.num_pixels = self.conf.get_int("train.num_pixels")
        self.plot_freq = self.conf.get_int("train.plot_freq")
        self.render_freq = self.conf.get_int("train.render_freq")
        self.checkpoint_freq = self.conf.get_int("train.checkpoint_freq", default=100)
        self.split_n_pixels = self.conf.get_int("train.split_n_pixels", default=10000)
        self.plot_conf = self.conf.get_config("plot")

        # logs
        self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, "logs"))

        # copy hparams to every module
        self.model.hparams = self.hparams
        self.loss.hparams = self.hparams

        # copyfile pointneus_disent file to exp_dir
        copyfile(f"{os.path.dirname(os.path.abspath(__file__))}/model/pointneus_disent.py", os.path.join(self.expdir, self.timestamp, 'pointneus_disent.py'))

    def load_from_dir(self, dir, checkpoint="latest"):
        """
        e.g. dir = exps_debug/ours_114/2022_10_21_12_46_36
        """
        old_checkpnts_dir = os.path.join(dir, "checkpoints")
        logger.info(f"Load from {old_checkpnts_dir} at {checkpoint}.pth")

        saved_model_state = torch.load(
            os.path.join(old_checkpnts_dir, "ModelParameters", str(checkpoint) + ".pth")
        )
        self.model.load_state_dict(saved_model_state["model_state_dict"])

        self.start_epoch = saved_model_state["epoch"]
        self.iter_step = saved_model_state["iter_step"]

        data = torch.load(
            os.path.join(
                old_checkpnts_dir, "OptimizerParameters", str(checkpoint) + ".pth"
            )
        )
        self.optimizer.load_state_dict(data["optimizer_state_dict"])

    def gen_plot_dataset(self):
        data_conf = copy.deepcopy(self.conf.get_config("dataset"))
        data_conf["img_res"] = [int(_ / 4.0) for _ in data_conf["img_res"]]
        if self.conf.get_string("dataset.data_dir") == 'dtu':
            self.plot_dataset = DTUDataset(**data_conf)
        elif self.conf.get_string("dataset.data_dir") == 'mipnerf':
            self.plot_dataset = MipDataset(**data_conf)
        elif self.conf.get_string("dataset.data_dir") == 'own_data':
            self.plot_dataset = OwnData(**data_conf)
        self.plot_dataloader = torch.utils.data.DataLoader(
            self.plot_dataset,
            batch_size=self.conf.get_int("plot.plot_nimgs"),
            shuffle=False,
            collate_fn=self.plot_dataset.collate_fn,
        )

    def gen_dataset(self, stg):
        data_conf_stg = self.data_confs[stg]
        if self.conf.get_string("dataset.data_dir") == 'dtu':
            self.train_dataset = DTUDataset(**data_conf_stg)
        elif self.conf.get_string("dataset.data_dir") == 'mipnerf':
            self.train_dataset = MipDataset(**data_conf_stg)
        elif self.conf.get_string("dataset.data_dir") == 'own_data':
            self.train_dataset = OwnData(**data_conf_stg)
        else:
            NotImplementedError

        logger.info(
            "Finish loading data. Data-set size: {0}".format(len(self.train_dataset))
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

        self.eval_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.train_dataset.collate_fn,
        )

        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.scale_factor = self.train_dataset.scale_factor
        self.n_batches = len(self.train_dataloader)

    def save_checkpoints(self, epoch, latest_only=False):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "iter_step": self.iter_step,
            },
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"),
        )
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(
                self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"
            ),
        )

        if latest_only:
            return 0

        print('SAVED WEIGHTS: ', epoch, self.iter_step)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "iter_step": self.iter_step,
            },
            os.path.join(
                self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"
            ),
        )
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(
                self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"
            ),
        )

    def train_step(self, batch, use_mvs=False, use_depth_reg=True):
        self.model.train()

        indices, model_input, ground_truth = batch
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()  # [1, 512, 2]
        model_input["pose"] = model_input["pose"].cuda()
        model_input["iter_step"] = self.iter_step
        
        # local data
        if model_input["local_data"] is not None:
            for key, value in model_input["local_data"].items():
                if isinstance(value, torch.Tensor):
                    model_input["local_data"][key] = value.to("cuda")

        fast = 1
        model_outputs = self.model(model_input, fast=fast)
        if use_mvs:
            model_outputs["pj"], model_outputs["pi"], _ = self.cost_mapping(
                z_vals=model_outputs["depth_vals"],
                ts=indices,
                xyz_raw=model_outputs["xyz"],
            )

        loss_output = self.loss(model_outputs, ground_truth)

        loss = loss_output["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        if self.hparams.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.on_after_backward()
        self.optimizer.step()
        self.scheduler.step()

        psnr = rend_util.get_psnr(
            model_outputs["rgb_values"], ground_truth["rgb"].cuda().reshape(-1, 3)
        )

        if self.total_step % 50 == 0:
            # write lr
            for i, param_group in enumerate(self.optimizer.param_groups):
                # print(f"Learning rate of group {i}: {param_group['lr']}")
                self.writer.add_scalar(f"lr/{i}", param_group['lr'], self.total_step)

            for k, v in loss_output.items():
                self.writer.add_scalar("t/" + k, v, self.total_step)

            self.writer.add_scalar(
                "t/beta", self.model.density.get_beta().item(), self.total_step
            )
            self.writer.add_scalar(
                "t/alpha", 1.0 / self.model.density.get_beta().item(), self.total_step
            )
            # only for neus
            # self.writer.add_scalar(
            #     "t/inv_s", self.model.deviation_network.get_variance().item(), self.total_step
            # )
            # self.writer.add_scalar(
            #     "t/s", 1.0 / self.model.deviation_network.get_variance().item(), self.total_step
            # )
            self.writer.add_scalar("t/psnr", psnr.item(), self.total_step)

        self.train_dataset.change_sampling_idx(self.num_pixels)

        self.iter_step += 1
        self.total_step += 1

    def render_step(self, batch, epoch, dataset, fast=-1):
        self.model.eval()

        indices, model_input, ground_truth = batch
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["pose"] = model_input["pose"].cuda()
        model_input["iter_step"] = self.iter_step

        # local data
        if model_input["local_data"] is not None:
            for key, value in model_input["local_data"].items():
                if isinstance(value, torch.Tensor):
                    model_input["local_data"][key] = value.to("cuda")

        split = utils.split_input(
            model_input, dataset.total_pixels, n_pixels=self.split_n_pixels
        )
        res = []
        for s in tqdm(split, leave=False):
            out = self.model(s, fast=fast)
            d = {
                "rgb_values": out["rgb_values"].detach().cpu(),
                "normal_map": out["normal_map"].detach().cpu(),
                "depth_values": out["depth_values"].detach().cpu(),
                "depth_vals": out["depth_vals"].detach().cpu(),
                "weights": out["weights"].detach().cpu(),
                "xyz": out["xyz"].detach().cpu(),
            }
            res.append(d)

        del out

        batch_size = ground_truth["rgb"].shape[0]
        model_outputs = utils.merge_output(res, dataset.total_pixels, batch_size)
        stack = []

        plot_data = self.get_plot_data(
            model_input, model_outputs, model_input["pose"], ground_truth["rgb"]
        )

        depth_cuda = plt.lin2img(plot_data["depth_map"][..., None], dataset.img_res)
        depth_cuda = depth_cuda[0].cuda() * self.scale_factor
        acc = model_outputs["weights"].sum(1).reshape(depth_cuda.shape)
        depth_cuda[acc < 0.2] = depth_cuda.max()

        mask = ground_truth["mask"].reshape(-1, 3)
        mask_bin = mask == 1.0
        mse = torch.mean(
            (model_outputs["rgb_values"] - ground_truth["rgb"].reshape(-1, 3))[mask_bin]
            ** 2
        )
        psnr = -10.0 * torch.log(mse) / torch.log(torch.Tensor([10.0]))
        self.writer.add_scalar("val/psnr", psnr.item(), self.total_step)

        stack += plt.stacked_plot(
            indices, plot_data, self.plots_dir, epoch, dataset.img_res, **self.plot_conf
        )
        stack[0][
            ~mask_bin.reshape(
                dataset.img_res
                + [
                    3,
                ]
            ).permute(2, 0, 1)
        ] = 0
        stack = torch.stack(stack, dim=0)  # (B, 3, H, W)
        self.writer.add_images("val/vis", stack, self.total_step)

        photo_conf = None

        self.total_step += 1

        return depth_cuda, photo_conf

    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs["rgb_values"].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs["normal_map"].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.0) / 2.0

        depth_map = model_outputs["depth_values"].reshape(batch_size, num_samples)
        acc = model_outputs["weights"].sum(1).reshape(batch_size, num_samples)

        plot_data = {
            "rgb_gt": rgb_gt,
            "pose": pose,
            "rgb_eval": rgb_eval,
            "normal_map": normal_map,
            "depth_map": depth_map,
            "acc": acc,
        }

        return plot_data


    def run(self, opt_stepN=1e5):
        # set everything here
        start_iter_step = self.iter_step
        logger.info(f"train spurfies at {self.checkpoints_path} ..")
        logger.info(
            f"[NOW] total_step={self.total_step}, iter_step={self.iter_step} stg={self.stg} opt_stepN={opt_stepN}"
        )

        # start pbar
        pbar = tqdm(total=opt_stepN - start_iter_step, desc="Train", ncols=60)

        # optimization: VolSDF uses 2000 epoch with 50 images, 2000*50 / 3 images -> ?
        epoch = self.start_epoch
        while True:
            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch)

            # render during training to check what's wrong
            early_render = (self.iter_step - start_iter_step <= 120 * 50) and epoch % (
                20 * 50 // self.ds_len
            ) == 0
            if epoch % self.render_freq == 0 or early_render:
                self.plot_dataset.change_sampling_idx(-1)
                self.plot_dataset.mode = "plot"
                batch = next(iter(self.plot_dataloader))
                self.plot_dataset.mode = "train"
                self.render_step(batch, epoch, self.plot_dataset, fast=-1)

                self.save_checkpoints(epoch, latest_only=True)
                torch.cuda.empty_cache()

            self.train_dataset.change_sampling_idx(self.num_pixels)
            for data_index, batch in enumerate(self.train_dataloader):
                self.train_step(batch, self.hparams.use_mvs)
                pbar.update(1)

            self.train_dataset.change_sampling_idx(-1)
             
            if self.iter_step - start_iter_step > opt_stepN:
                break

            epoch += 1
        logger.info(f"final_epoch : {epoch}")
        # close pbar
        pbar.close()

        # save
        self.save_checkpoints(epoch)
        self.start_epoch = epoch

        return epoch

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )

                if not valid_gradients:
                    print(name)
                    break

        if not valid_gradients:
            logger.warning(
                f"detected inf or nan values in gradients. not updating model parameters"
            )
            self.optimizer.zero_grad()
