import argparse
import GPUtil
import os
import gc
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity

import spurfies.utils.general as utils
import spurfies.utils.plots as plt
from helpers.help import logger

torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

from lpipsPyTorch import lpips
os.makedirs('./lpipsPyTorch/weights/', exist_ok=True)
torch.hub.set_dir('./lpipsPyTorch/weights/')

def evaluate(**kwargs):
    conf = ConfigFactory.parse_file(kwargs["conf"])
    exps_folder_name = kwargs["exps_folder_name"]
    evals_folder_name = kwargs["evals_folder_name"]

    root = "./"
    expname = kwargs["expname"]
 
    dataset_conf = conf.get_config("dataset")
    if dataset_conf.get_string('data_dir') == 'dtu':
        scan_id = (
            kwargs["scan_id"]
            if kwargs["scan_id"] != -1
            else conf.get_int("dataset.scan_id", default=-1)
        )
    else:
        scan_id = kwargs["scan_id"]

    if scan_id != -1:
        expname = expname + "_{0}".format(scan_id)
    else:
        scan_id = conf.get_string("dataset.object", default="")

    if kwargs["ckpt_dir"] == "" and kwargs["timestamp"] == "latest":
        print(os.path.join(root, kwargs["exps_folder_name"], expname))
        if os.path.exists(os.path.join(root, kwargs["exps_folder_name"], expname)):
            timestamps = os.listdir(
                os.path.join(root, kwargs["exps_folder_name"], expname)
            )
            if (len(timestamps)) == 0:
                print("WRONG EXP FOLDER")
                exit()

            timestamp = None
            for t in sorted(timestamps):
                if os.path.exists(
                    os.path.join(
                        root,
                        kwargs["exps_folder_name"],
                        expname,
                        t,
                        "checkpoints",
                        "ModelParameters",
                        str(kwargs["checkpoint"]) + ".pth",
                    )
                ):
                    timestamp = t
            if timestamp is None:
                print("NO GOOD TIMSTAMP")
                exit()
        else:
            print("WRONG EXP FOLDER")
            exit()
    else:
        timestamp = kwargs["timestamp"]

    utils.mkdir_ifnotexists(os.path.join(root, evals_folder_name))
    expdir = os.path.join(root, exps_folder_name, expname)
    evaldir = os.path.join(root, evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config("dataset")
    if kwargs["scan_id"] != -1:
        dataset_conf["scan_id"] = kwargs["scan_id"]
    dataset_conf["num_views"] = -1  # all images
    dataset_conf["data_dir_root"] = opt.data_dir_root
    print('eval dataset_conf: ', dataset_conf)
    dataset_conf["mode"] = 'eval'
    eval_dataset = utils.get_class(conf.get_string("train.dataset_class"))(
        **dataset_conf
    )

    conf_model = conf.get_config("model")
    model = utils.get_class(conf.get_string("train.model_class"))(conf=conf_model, scan_id=kwargs["scan_id"], dataset=dataset_conf.get_string('data_dir'))

    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()

    if opt.eval_rendering:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
        )
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res
        logger.info(f"img res = {img_res}")

    if kwargs["ckpt_dir"] != "":
        old_checkpnts_dir = os.path.join(root, kwargs["ckpt_dir"], "checkpoints")
    else:
        old_checkpnts_dir = os.path.join(expdir, timestamp, "checkpoints")
    
    logger.info(f"load model from: {old_checkpnts_dir}")
    if opt.result_from != "None":
        epoch = 0
        # use the latest epoch's rendering results
        for renderdir in os.listdir(evaldir):
            if renderdir.startswith("rendering_"):
                epoch = max(epoch, int(renderdir.replace("rendering_", "")))
    else:
        saved_model_state = torch.load(
            os.path.join(
                old_checkpnts_dir, "ModelParameters", str(kwargs["checkpoint"]) + ".pth"
            )
        )
        print(model)
        model.load_state_dict(saved_model_state["model_state_dict"])
        epoch = saved_model_state["epoch"]

    ####################################################################################################################
    model.cuda()
    model.eval()

    if opt.eval_mesh:
        with torch.no_grad():
            if "dtu" in opt.conf:
                bb_dict = np.load(os.path.join(opt.data_dir_root, "dtu/bbs.npz"))
                if scan_id == 82:
                    grid_params = bb_dict[str(83)]
                elif scan_id in [21, 34, 38]:
                    grid_params = bb_dict[str(24)]
                else:
                    grid_params = bb_dict[str(scan_id)]

                mesh = plt.get_surface_by_grid(
                    grid_params=grid_params,
                    sdf=lambda x: model.get_sdf_eval(x),
                    resolution=kwargs["resolution"],
                    level=conf.get_int("plot.level", default=0),
                    higher_res=False,
                )
            elif "mip_nerf" in opt.conf:
                if scan_id == 'garden':
                    # garden pts bounds
                    grid_params = np.array([[-1.98553513, -1.56664338, -1.62574293],
                                            [ 1.85141105,  0.89619044,  2]])
                elif scan_id == 'stump':
                    # stump pts bounds
                    grid_params = np.array([[-1.94276784, -1.27124258, -1.18201152],
                                            [2., 1.41121731,  1.30076323]])
                else:
                    raise NotImplementedError

                mesh = plt.get_surface_by_grid(
                    grid_params=grid_params,
                    sdf=lambda x: model.get_sdf_eval(x),
                    resolution=kwargs["resolution"],
                    level=conf.get_int("plot.level", default=0),
                    higher_res=False,
                )
            else:
                raise NotImplementedError

            # Transform to world coordinates
            mesh.apply_transform(scale_mat)

            # Taking the biggest connected component
            components = mesh.split(only_watertight=False)
            areas = np.array([c.area for c in components], dtype=np.float32)
            mesh_clean = components[areas.argmax()]

            mesh_folder = "{0}/mesh_{1}".format(evaldir, epoch)
            utils.mkdir_ifnotexists(mesh_folder)
            mesh_clean.export("{0}/scan{1}.ply".format(mesh_folder, scan_id), "ply")

            del mesh_clean, components, areas, mesh
            gc.collect()
            torch.cuda.empty_cache()

    if opt.eval_rendering:
        images_dir = "{0}/rendering_{1}".format(evaldir, epoch)
        logger.info(f"rendered images dir: {images_dir}")
        utils.mkdir_ifnotexists(images_dir)
        os.makedirs(os.path.join(images_dir, "depth_est"), exist_ok=True)

        if "dtu" in opt.conf:
            from spurfies.datasets.dtu import get_eval_ids
            test_idx = get_eval_ids("DTU", scan_id=None)
        elif "mip_nerf" in opt.conf:
            from spurfies.datasets.mip_nerf import get_eval_ids
            test_idx = get_eval_ids("mip_nerf", scan_id, mode='eval')
            print('mip nerf: ', test_idx)
        else:
            raise NotImplementedError

        logger.info(f"{len(test_idx)} images")

        psnrs, ssims, lpipss = [], [], []
        for data_index, (indices, model_input, ground_truth) in enumerate(
            eval_dataloader
        ):
            if indices not in test_idx:
                continue

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["pose"] = model_input["pose"].cuda()

            # Already have results -> evaluate
            if opt.result_from != "None":

                pred_img = Image.open(
                        "{0}/eval_{1}.png".format(images_dir, "%03d" % indices[0])
                    )

                rgb_pred = np.array(pred_img, dtype=np.float32).reshape(-1, 3) / 255.0
                rgb_pred = torch.from_numpy(rgb_pred)
                mask = ground_truth["mask"].reshape(-1, 3)
                mask_bin = mask == 1.0
                rgb_fg = ground_truth["rgb"].reshape(-1, 3) * mask + (1 - mask)
                rgb_hat_fg = rgb_pred * mask + (1 - mask)
                rgb_fg = (
                    rgb_fg.reshape(
                        img_res
                        + [
                            3,
                        ]
                    )
                    .cpu()
                    .numpy()
                )  # (HW, 3) -> (H, W, 3)
                rgb_hat_fg = (
                    rgb_hat_fg.reshape(
                        img_res
                        + [
                            3,
                        ]
                    )
                    .cpu()
                    .numpy()
                )
                mse = torch.mean(
                    (rgb_pred - ground_truth["rgb"].reshape(-1, 3))[mask_bin] ** 2
                )
                psnr_masked = -10.0 * torch.log(mse) / torch.log(torch.Tensor([10.0]))
                ssim_masked = float(
                    structural_similarity(rgb_hat_fg, rgb_fg, data_range=1, multichannel=True, channel_axis=2)
                )
                
                rgb_hat_fg_lp = torch.from_numpy(rgb_hat_fg).cuda().permute(2, 0, 1)[None, ...]
                rgb_fg_lp = torch.from_numpy(rgb_fg).cuda().permute(2, 0, 1)[None, ...]
                lpips_masked = lpips(rgb_hat_fg_lp, rgb_fg_lp, net_type='vgg').detach().cpu().numpy()[0]

                ssims.append(ssim_masked)
                psnrs.append(psnr_masked)
                lpipss.append(lpips_masked)

            # Otherwise, render and save RGBD for evaluation
            else:
                split = utils.split_input(
                    model_input, total_pixels, n_pixels=opt.split_n_pixels
                )
                res = []
                for s in tqdm(split, ncols=60):
                    out = model(s)
                    res.append(
                        {
                            "rgb_values": out["rgb_values"].detach(),
                            "normal_map": out["normal_map"].detach(),
                            "depth_values": out["depth_values"].detach(),
                            "weights": out["weights"].detach(),
                        }
                    )

                batch_size = ground_truth["rgb"].shape[0]
                model_outputs = utils.merge_output(res, total_pixels, batch_size)

                # RGB
                rgb_eval = model_outputs["rgb_values"]
                rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
                rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
                rgb_eval = rgb_eval.transpose(1, 2, 0)

                # Depth
                depth_map = model_outputs["depth_values"].reshape(
                    batch_size, total_pixels
                )
                depth_map = (
                    plt.lin2img(depth_map[..., None], img_res).detach().cpu().squeeze()
                )
                # save depth maps for image-based rendering
                depth_map_np = (
                    depth_map.numpy() * eval_dataset.scale_factor
                )  # (576, 768) float32

                # Save in png
                ## RGB
                img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
                img.save("{0}/eval_{1}.png".format(images_dir, "%03d" % indices[0]))

                ## Normal
                normal_eval = model_outputs["normal_map"]
                normal_eval = normal_eval.reshape(batch_size, total_pixels, 3)
                normal_eval = (normal_eval + 1.0) / 2.0
                normal_eval = (
                    plt.lin2img(normal_eval, img_res).detach().cpu().numpy()[0]
                )
                normal_eval = normal_eval.transpose(1, 2, 0)
                normal_eval = Image.fromarray((normal_eval * 255).astype(np.uint8))
                normal_eval.save(
                    "{0}/normal_{1}.png".format(images_dir, "%03d" % indices[0])
                )

                ## Depth
                acc = model_outputs["weights"].sum(1).reshape(batch_size, total_pixels)
                acc = (
                    plt.lin2img(acc[..., None], img_res)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                depth_map = plt.visualize_depth(depth_map, acc)
                depth_map = Image.fromarray((depth_map * 255).astype(np.uint8))
                depth_map.save(
                    "{0}/dep_{1}.png".format(images_dir, "%03d" % indices[0])
                )

                torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        if opt.result_from != "None":
            psnrs = np.array(psnrs).astype(np.float64)
            ssims = np.array(ssims).astype(np.float64)
            lpipss = np.array(lpipss).astype(np.float64)
            print(f"SCAN {scan_id}:")
            print(
                "    psnr mean = {0}, std {1}".format(
                    "%.4f" % psnrs.mean(), "%.4f" % psnrs.std()
                )
            )
            print(
                "    ssim mean = {0}, std {1}".format(
                    "%.4f" % ssims.mean(), "%.4f" % ssims.std()
                )
            )
            print(
                "    lpips mean = {0}, std {1}".format(
                    "%.4f" % lpipss.mean(), "%.4f" % lpipss.std()
                )
            )

            return psnrs, ssims, lpipss

    return np.array([0]), np.array([0]), np.array([0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="dtu")
    parser.add_argument(
        "--data_dir_root", type=str, default="data", help="GT data dir"
    )
    parser.add_argument(
        "--eval_mesh",
        default=False,
        action="store_true",
        help="extract mesh via marching cube",
    )
    parser.add_argument(
        "--eval_rendering",
        default=False,
        action="store_true",
        help="If set, evaluate rendering quality.",
    )
    parser.add_argument(
        "--result_from", default="None", type=str, choices=["None", "default", "blend"]
    )
    parser.add_argument(
        "--expname",
        type=str,
        default="ours",
        help="The experiment name to be evaluated.",
    )
    parser.add_argument(
        "--exps_folder",
        type=str,
        default="exps_vsdf",
        help="The experiments folder name for train.",
    )
    parser.add_argument(
        "--evals_folder",
        type=str,
        default="exps_result",
        help="The evaluation folder name (a new folder).",
    )
    parser.add_argument("--gpu", type=str, default="auto", help="GPU to use")
    parser.add_argument(
        "--timestamp",
        default="latest",
        type=str,
        help="The experiemnt timestamp to test.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        type=str,
        help="The trained model checkpoint to test",
    )
    parser.add_argument("--ckpt_dir", default="", type=str)
    parser.add_argument(
        "--scan_ids", nargs="+", type=int, default=None, help="e.g. --scan_ids 12 34 56"
    )
    parser.add_argument(
        "--resolution",
        default=512,
        type=int,
        help="Grid resolution for marching cube, set as 400 if not enough GPU",
    )
    parser.add_argument("--split_n_pixels", default=512, type=int)
    opt = parser.parse_args()

    # configs
    opt.conf = f"./config/confs/{opt.conf}.conf"
    opt.eval_rendering = (opt.result_from != "None") or opt.eval_rendering
    if opt.scan_ids is None:
        if "dtu" in opt.conf:
            opt.scan_ids = [21, 24, 34, 37, 38, 40, 82, 106, 110, 114, 118]
        elif "mip_nerf" in opt.conf:
            opt.scan_ids = ['garden', 'stump']
        else:
            raise NotImplementedError
        
    # GPU
    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(
            order="memory",
            limit=1,
            maxLoad=0.9,
            maxMemory=0.9,
            includeNan=False,
            excludeID=[],
            excludeUUID=[],
        )
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(gpu)

    # eval
    res = {}
    psnr_all, ssim_all, lpips_all = [], [], []
    for scan_id in opt.scan_ids:
        logger.info(f"scan_id = {scan_id}")
        exps_folder = os.path.join(opt.exps_folder, str(scan_id))
        evals_folder = os.path.join(opt.evals_folder, str(scan_id))

        # exps_folder = os.path.join(opt.exps_folder) #, str(scan_id))
        # evals_folder = os.path.join(opt.evals_folder) #, str(scan_id))
        print(exps_folder, evals_folder)
        assert opt.result_from in ["None", "default"]
        assert os.path.exists(opt.exps_folder)
        logger.warning(f"result folder = {opt.evals_folder}")
        logger.warning(f"volsdf ckpt folder = {opt.exps_folder}/{opt.expname}")
        logger.info(f"scan_ids = {opt.scan_ids}")

        psnr_i, ssim_i, lpips_i = evaluate(
            conf=opt.conf,
            expname=opt.expname,
            exps_folder_name=exps_folder,
            evals_folder_name=evals_folder,
            timestamp=opt.timestamp,
            checkpoint=opt.checkpoint,
            scan_id=int(scan_id),
            resolution=opt.resolution,
            ckpt_dir=opt.ckpt_dir,
        )
        res[scan_id] = {}
        res[scan_id]["psnr"] = psnr_i.mean().tolist()
        res[scan_id]["ssim"] = ssim_i.mean().tolist()
        res[scan_id]["lpips"] = lpips_i.mean().tolist()

        psnr_all.append(psnr_i.tolist())
        ssim_all.append(ssim_i.tolist())
        lpips_all.append(lpips_i.tolist())

    if opt.result_from != "None":
        print("FINAL metric: ")
        print(f"    psnr = {np.mean([np.mean(_) for _ in psnr_all]):.4f}")
        print(f"    ssim = {np.mean([np.mean(_) for _ in ssim_all]):.4f}")
        print(f"    lpips = {np.mean([np.mean(_) for _ in lpips_all]):.4f}")

    res["psnr_mean"] = np.mean([np.mean(_) for _ in psnr_all])
    res["ssim_mean"] = np.mean([np.mean(_) for _ in ssim_all])
    res["lpips_mean"] = np.mean([np.mean(_) for _ in lpips_all])