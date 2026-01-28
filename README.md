# 3D Equivariant Visuomotor Policy Learning via Spherical Projection

[Project Website](https://isp-3d.github.io/) | [Paper](https://arxiv.org/pdf/2505.16969) | [Video](https://youtu.be/b1aAnbDHQh0?si=DGoHkV6DgSMSa7VS)  

<a href="https://bocehu.github.io/">Boce Hu</a><sup>1</sup>,
<a href="https://www.dianwang.io/">Dian Wang</a><sup>2</sup>,
<a href="https://dmklee.github.io/">David Klee</a><sup>1</sup>,
<a href="https://heng-tian.github.io/">Heng Tian</a><sup>1</sup>,
<a href="https://zxp-s-works.github.io/">Xupeng Zhu</a><sup>1</sup>,
<a href="https://haojhuang.github.io/">Haojie Huang</a><sup>1</sup>,<br>
<a href="https://www2.ccs.neu.edu/research/helpinghands/people/">Robert
    Platt<sup>&dagger;1</sup></a>,
<a href="https://www.robinwalters.com/">Robin Walters<sup>&dagger;1</sup></a>

<sup>1</sup>Northeastern University, <sup>2</sup>Stanford University

NeurIPS 2025 (Spotlight)
![](img/method_pipeline.png)

## Installation
1.  Install the following apt packages for mujoco:
    ```bash
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    ```
1. Install gfortran (dependency for escnn) 
    ```bash
    sudo apt install -y gfortran
    ```

1. Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)

1. Clone this repo
    ```bash
    git clone https://github.com/BoceHu/ISP.git
    cd ISP
    ```

1. Install environment:
    ```bash
    mamba env create -f conda_environment.yaml
    conda activate isp
    ```
    
    If you are using a Blackwell-architecture GPU (e.g., RTX 5090), please follow the steps below:
    ```
    mamba env create -f env_blackwell.yaml
    conda activate isp
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
    ```
    This will create the appropriate environment and build PyTorch3D from source, which is required for Blackwell GPUs.

1. Install mimicgen:
    ```bash
    cd ..
    git clone https://github.com/NVlabs/mimicgen_environments.git
    cd mimicgen_environments
    git checkout 45db4b35a5a79e82ca8a70ce1321f855498ca82c
    pip install -e .
    cd ../ISP
    ```
1. Make sure mujoco version is 2.3.2 (required by mimicgen)
    ```bash
    pip list | grep mujoco
    ```

## Dataset
### Download Dataset
Please visit the link below to download the datasets.

 https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core

Make sure the dataset is kept under `/path/to/ISP/data/robomimic/datasets/[dataset]/[dataset].hdf5`

### Generating a larger FOV observation

```bash
# Template
python isp/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/[dataset]/[dataset].hdf5 --output data/robomimic/datasets/[dataset]/[dataset]_fisheye.hdf5 --num_workers=[n_worker]

# Replace [dataset] and [n_worker] with your choices.
# E.g., use 24 workers to generate point cloud and voxel observation for stack_d1

python isp/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/stack_d1/stack_d1.hdf5 --output data/robomimic/datasets/stack_d1/stack_d1_fisheye.hdf5 --num_workers=16
```

### Convert Action Space in Dataset
The downloaded dataset has a relative action space. To train with an absolute action space, the dataset needs to be converted accordingly
```bash
# Template
python isp/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/[dataset]/[dataset]_fisheye.hdf5 -o data/robomimic/datasets/[dataset]/[dataset]_fisheye_abs.hdf5 -n [n_worker]

# Replace [dataset] and [n_worker] with your choices.
# E.g., convert stack_d1_fisheye with 16 workers

python isp/scripts/robomimic_dataset_conversion.py -i data/robomimic/datasets/stack_d1/stack_d1_fisheye.hdf5 -o data/robomimic/datasets/stack_d1/stack_d1_fisheye_abs.hdf5 -n 16
```

put the processed dataset under `ISP/data/robomimic/datasets/[task_name]/`

e.g. `ISP/data/robomimic/datasets/stack_d1/stack_d1_fisheye_abs.hdf5`

## Training
To train our method on Stack D1 task:


### SO(2) version
```bash
python train.py --config-name=train_isp_so2 task_name=stack_d1 n_demo=100
```
### SO(3) version
```bash
python train.py --config-name=train_isp_so3 task_name=stack_d1 n_demo=100
```
### SO(2) pretrained encoder
```bash
python train.py --config-name=train_isp_so2_pretrain task_name=stack_d1 n_demo=100
```

We recommend starting with the SO(2) version, as it is faster and more GPU-friendly.

To train on other tasks, replace `stack_d1` with one of the following: `stack_three_d1`, `square_d2`, `threading_d0`, `coffee_d2`, `three_piece_assembly_d0`, `hammer_cleanup_d1`, `mug_cleanup_d1`, `kitchen_d1`, `nut_assembly_d0`, `pick_place_d0`, `coffee_preparation_d1`. Please ensure that the corresponding dataset has been downloaded in advance.

To run environments on CPU (to save GPU memory), use `osmesa` instead of `egl` through `MUJOCO_GL=osmesa PYOPENGL_PLATTFORM=osmesa`, e.g.,
```bash
MUJOCO_GL=osmesa PYOPENGL_PLATTFORM=osmesa python train.py --config-name=train_isp_so3 task_name=stack_d1
```
Note that this will take longer to roll out the policy.

For inference, the SO(3) model itself requires approximately 16 GB of GPU memory with the default batch size of 64, while the SO(2) variant requires around 10 GB under the same settings.

If you want to assign both the environments and the policy to a specific GPU, you can do the following:
```bash
EGL_DEVICE_ID=1 MUJOCO_EGL_DEVICE_ID=1 HYDRA_FULL_ERROR=1 python train.py --config-name=train_isp_so2_pretrain task_
name=stack_d1 n_demo=100 training.device=1
```
``EGL_DEVICE_ID`` and ``MUJOCO_EGL_DEVICE_ID`` control which GPU is used for MuJoCo rendering, while ``training.device`` specifies the GPU used for policy training.


## 📜 Citation
```bibtex
@article{hu20253d,
  title   = {3D Equivariant Visuomotor Policy Learning via Spherical Projection},
  author  = {Hu, Boce and
             Wang, Dian and
             Klee, David and
             Tian, Heng and
             Zhu, Xupeng and
             Huang, Haojie and
             Platt, Robert and
             Walters, Robin},
  journal = {arXiv preprint arXiv:2505.16969},
  year    = {2025}
}
```

## License
This project is released under the Academic Research and Educational Use License. The software is intended for academic research and educational purposes only.
Commercial use is strictly prohibited without prior written permission from the authors. 
Please see the [LICENSE](LICENSE) file for the full license text and detailed terms.


## Acknowledgement
* Our repo is built upon the original [Equivariant Diffusion Policy](https://github.com/pointW/equidiff).
* Our Diffusion Policy baseline is adapted from the codebase of [Diffusion Policy](https://github.com/real-stanford/diffusion_policy).
* Our ACT baseline is adapted from its [original repo](https://github.com/tonyzhaozh/act).
