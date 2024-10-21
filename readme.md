`Spurfies: Sparse Surface Reconstruction using Local Geometry Priors`  
[[Project page](https://geometric-rl.mpi-inf.mpg.de/spurfies/)], [[arxiv](https://arxiv.org/abs/2408.16544)]

![Example of some reconstruction](assets/teaser.png)

## Installation

1. Clone spurfies and set up env.
```bash
# clone repo
git clone 

# create env
conda create -n custom_name python=3.11 cmake=3.14.0

# install torch
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

pip install requirements.txt

# install torch_knnquery for all cuda architectures
cd torch_knnquery
CXX=g++-11 CC=gcc-11 TORCH_CUDA_ARCH_LIST="6.0 7.0 7.5 8.0 8.6+PTX" python -m pip install .
```

## Data and Checkpoints

https://github.com/naver/dust3r/tree/main?tab=readme-ov-file#checkpoints

place 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth' inside 'dust3r/checkpoints' path

spurfies: https://drive.google.com/drive/folders/1xp17J_DPpr4dJ6NgEe76n1UGOZSJtcyC?usp=sharing

copy the files correspondingly to data/ and ckpt/ folder in project dir

## Usage

1. Get Neural Points using DUSt3R from known camera pose:
```bash
# dtu
python dust3r_inference.py --views 3 --dataset dtu --scan_id [21,24,34,37,38,40,82,106,110,114,118]

# mipnerf:
python dust3r_inference.py --views 3 --dataset mipnerf --scan_id [garden,stump]
```

2. Training spurfies:
```bash
# dtu 24
python runner.py testlist=scan24 vol=dtu_pn outdir=results/dtu/24   exps_folder=results/dtu/24 opt_stepNs=[100_000,0,0]

# mipnerf garden
python runner.py testlist=garden vol=dtu_pn outdir=results/mipnerf/garden   exps_folder=results/mipnerf/garden opt_stepNs=[100_000,0,0]
```
3. Rendering NVS and Mesh
```bash
# dtu 24
python eval_vsdf.py --conf dtu_pn --data_dir_root data --scan_ids 24 --gpu 0 --expname ours --exps_folder results/dtu/ --evals_folder results/dtu/ --eval_mesh --eval_rendering

# mipnerf garden
python eval_vsdf.py --conf mipnerf --data_dir_root data --scan_ids garden --gpu 0 --expname ours --exps_folder results/mipnerf/ --evals_folder results/mipnerf/ --eval_mesh --eval_rendering
```

4. Evaluate Mesh
```bash
# dtu 24
python evals/eval_dtu.py --datadir dtu --scan 24 --data_dir_root data
```
5. Evaluate NVS
```bash
# dtu all scans
python eval_vsdf.py --conf dtu_pn --data_dir_root data --eval_rendering --expname ours --exps_folder results/dtu/ --evals_folder results/dtu/ --result_from default
```

## Credits
Code built upon [S-VolSDF](https://hao-yu-wu.github.io/s-volsdf/)

```bibtex
@article{raj2024spurfies,
  title={Spurfies: Sparse Surface Reconstruction using Local Geometry Priors},
  author={Raj, Kevin and Wewer, Christopher and Yunus, Raza and Ilg, Eddy and Lenssen, Jan Eric},
  journal={arXiv preprint arXiv:2408.16544},
  year={2024}
}
```