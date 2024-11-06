## Spurfies

Implementation of **Spurfies: Sparse Surface Reconstruction using Local Geometry Priors**

[[Project page](https://geometric-rl.mpi-inf.mpg.de/spurfies/)] | [[arxiv](https://arxiv.org/abs/2408.16544)] | 
[[data & ckpt](https://drive.google.com/drive/folders/1xp17J_DPpr4dJ6NgEe76n1UGOZSJtcyC?usp=sharing)]

![Example of some reconstruction](assets/teaser.png)

## Installation
The code is compatible with python-3.11, torch-2.0, and cuda-11.8

1. Clone spurfies and set up env.
```bash
# clone repo
git clone https://github.com/kevinYitshak/spurfies.git
git submodule update --init --recursive
mv dust3r_inferfence.py dust3r
mv dust3r_inferfence_own.py dust3r

# create env
conda create -n *custom_name* python=3.11 cmake=3.14.0

# install torch
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# install other requirements
pip install requirements.txt

# install torch_knnquery for all cuda architectures
cd torch_knnquery
CXX=g++-11 CC=gcc-11 TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" python -m pip install .
```

## Data and Checkpoints

1. **DUSt3R checkpoints** from offical repo: 
https://github.com/naver/dust3r/tree/main?tab=readme-ov-file#checkpoints

    place 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth' inside 'dust3r/checkpoints' path

2. **Spurfies** checkpoints and data: https://drive.google.com/drive/folders/1xp17J_DPpr4dJ6NgEe76n1UGOZSJtcyC?usp=sharing

   - copy the files correspondingly to data/ and ckpt/ folder in project dir
   - **checkpoints:**
       - local_prior.pt: trained local geometry prior on ShapeNet data
       - vismvsnet.pt: used for feature consistency loss [Vis-MVSNet](https://github.com/jzhangbs/Vis-MVSNet) 
   - **data:** contains dtu and mipnerf datasets

## Usage on own data
1. Place your images in the 'data/own_data/scene_name/image'.
```bash
cd dust3r
python dust3r_inference_own.py --views 3 --dataset own_data --scan_id 'scene_name' 
```
This will save the resized images (512x384), camera pose (opencv format), and the pointcloud in ``/data/own_data/scene_name`` folder obtained from DUSt3R, which we use it to train Spurfies. Note: The quality of the reconstruction depends upon the pointcloud obtained from DUSt3R. **Replace 'scene_name' with your own**

An example scene named 'duck' is given.

2.  Optimizing geometry, color latent codes and color network:
```bash
python runner.py testlist=scene_name vol=own_data outdir=results/own_data/scene_name exps_folder=results/own_data/scene_name opt_stepNs=[100_000,0,0]
```

## Usage

1. Get Neural Points using DUSt3R from known camera pose **(already provided in data)**:
```bash
cd dust3r
# dtu
python dust3r_inference.py --views 3 --dataset dtu --scan_id [21,24,34,37,38,40,82,106,110,114,118]

# mipnerf:
python dust3r_inference.py --views 3 --dataset mipnerf --scan_id [garden,stump]
```

2. Optimizing geometry, color latent codes and color network:

In this stage, the geometry latent code, color latent code, and color network is optimized using differentiable volume rendering. The geometry network is frozen with local geometry prior.
```bash
# dtu 24
python runner.py testlist=scan24 vol=dtu_pn outdir=results/dtu/24   exps_folder=results/dtu/24 opt_stepNs=[100_000,0,0]

# mipnerf garden
python runner.py testlist=garden vol=mip_nerf outdir=results/mipnerf/garden   exps_folder=results/mipnerf/garden opt_stepNs=[100_000,0,0]
```
3. Rendering NVS and Mesh
```bash
# dtu 24
python eval_spurfies.py --conf dtu_pn --data_dir_root data --scan_ids 24 --gpu 0 --expname ours --exps_folder results/dtu/ --evals_folder results/dtu/ --eval_mesh --eval_rendering

# mipnerf garden
python eval_spurfies.py --conf mipnerf --data_dir_root data --scan_ids garden --gpu 0 --expname ours --exps_folder results/mipnerf/ --evals_folder results/mipnerf/ --eval_mesh --eval_rendering
```

4. Evaluate Mesh
```bash
# dtu
python evals/eval_dtu.py --datadir dtu --scan -1 --data_dir_root data
```
5. Evaluate NVS
```bash
# dtu all scans
python eval_spurfies.py --conf dtu_pn --data_dir_root data --eval_rendering --expname ours --exps_folder results/dtu/ --evals_folder results/dtu/ --result_from default
```

## Credits
Code built upon:
-  [S-VolSDF: Sparse Multi-View Stereo Regularization of Neural Implicit Surfaces](https://hao-yu-wu.github.io/s-volsdf/)
```bibtex
@article{wu2023s,
  title={S-VolSDF: Sparse Multi-View Stereo Regularization of Neural Implicit Surfaces},
  author={Wu, Haoyu and Graikos, Alexandros and Samaras, Dimitris},
  journal={ICCV},
  year={2023}
}
```

## Citation
```bibtex
@article{raj2024spurfies,
  title={Spurfies: Sparse Surface Reconstruction using Local Geometry Priors},
  author={Raj, Kevin and Wewer, Christopher and Yunus, Raza and Ilg, Eddy and Lenssen, Jan Eric},
  journal={arXiv preprint arXiv:2408.16544},
  year={2024}
}
```