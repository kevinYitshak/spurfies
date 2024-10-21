#--------------------------args & GPU-------------------------------------#
from omegaconf import DictConfig, OmegaConf
import hydra
from helpers.help import logger
from helpers.help import run_help

# init args
@hydra.main(version_base=None, config_path="config", config_name="ours")
def get_config(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    global args
    print(cfg)
    args = cfg
get_config()

# update args & set GPU
args = run_help(args)
#------------------------------------------------------------------------#

import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from helpers.utils import *
from spurfies.train import VolOpt

cudnn.benchmark = True
torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True, precision=4)


def save_scene_depth(scene):
    torch.cuda.empty_cache()

    # volume optimizer
    vol_opt = VolOpt(args=args,
                    batch_size=1,
                    is_continue=args.get('is_continue', False),
                    timestamp='latest',
                    checkpoint='latest',
                    scan=scene)

    #-----------------------------------------------------------#
    stage_idx = 0
    vol_opt.gen_dataset(stage_idx)
    vol_opt.stg = stage_idx
    # vol_opt.loss.set_stg(stage_idx)

    epoch = vol_opt.run(args.opt_stepNs[stage_idx])
    print('finished training')

if __name__ == '__main__':
    if 'txt' in args.testlist:
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        testlist = [x for x in args.testlist.replace(' ', '').split(',') if x]

    logger.warning(f"{testlist} {args.outdir} {args.exps_folder}")

    for scene in testlist:
        save_scene_depth(scene)
    