<p align="center">
  <h2 align="center">[ICRA 2026] PROFusion: Robust and Accurate Dense Reconstruction via Camera Pose Regression and Optimization</h2>
  <p align="center">
    <a href="https://siyandong.github.io/">Siyan Dong</a>
    ·
    <a href="TODO">Zijun Wang</a>
    ·
    <a href="TODO">Lulu Cai</a>
    ·
    <a href="https://people.eecs.berkeley.edu/~yima/">Yi Ma</a>
    ·
    <a href="https://yanchaoyang.github.io/">Yanchao Yang</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2509.24236">Paper</a> | <a href="TODO">Video</a> | <a href="TODO">Poster</a> </h3>
<p align="center">
 
<div align="center">
  <img src="./assets/shake_and_fast.gif" width="99%" /> 
  <img src="./assets/p207.gif" width="99%" /> 
</div>


A simple yet effective system for real-time camera tracking and dense scene reconstruction, providing both robustness against unstable camera motions and accurate reconstruction results.


## Installation

1. Clone PROFusion
```bash
git clone https://github.com/siyandong/PROFusion.git
cd PROFusion
```

2. Prepare environment
```bash
conda create -n profusion python=3.11 cmake=3.14 -y
conda activate profusion

# install torch according to your cuda version, e.g., 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# install packages required by the pose regression module
pip install -r requirements.txt

# install packages required by the optimization and fusion module
conda install -y -c conda-forge gcc_linux-64=11.4.0 gxx_linux-64=11.4.0 pybind11 
```
Install Eigen and OpenCV with CUDA required by the optimization and fusion module, following [this link](TODO)

3. Build PROFusion: Set sm_xx and compute_xx in [L30 in CMakeLists.txt](TODO) according to your architecture and run the following script
```
bash build.sh
```

4. Optional: If you cannot load the pretrained model automatically, [manually download](https://huggingface.co/siyan824/profusion_pr/blob/main/profusion_pr.pth) and format the directory as shown in [L11 in pose_regression/run.py](TODO)

5. Optional: Acceleration
```bash
# compile cuda kernels for RoPE
cd pose_regression/modules/pos_embed/curope/
python setup.py build_ext --inplace
cd ../../../../
```


## Run demos

Download example data from [Google Drive](https://drive.google.com/drive/folders/1bNKpEORq88b85XLAQ7RwzO_plhB_jqmO?usp=sharing) and format directories as: 
```
data/
├── femtobolt/
├── ├── p207/
├── ├── ├── color/
├── ├── ├── depth/
├── ├── p302_sparse/
├── fastcamo/example/
└── eth3d/cs1/
```
Otherwise, you can edit these paths in configs/data_config/xx.yaml

#### FemtoBolt
```
python run_profusion.py configs/data_config/femtobolt.yaml
```

#### FastCaMo-Synth (noise)
```
python run_profusion.py configs/data_config/fastcamo.yaml
```

#### ETH3D
```
python run_profusion.py configs/data_config/eth3d.yaml
```


## Citation

If you find our work helpful in your research, please consider citing: 
```
@article{dong2025profusion,
  title={PROFusion: Robust and Accurate Dense Reconstruction via Camera Pose Regression and Optimization},
  author={Dong, Siyan and Wang, Zijun and Cai, Lulu and Ma, Yi and Yang, Yanchao},
  journal={arXiv preprint arXiv:2509.24236},
  year={2025}
}
```


## Acknowledgments

Our implementation is based on: [DUSt3R](https://github.com/naver/dust3r), [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R), [Reloc3r](https://github.com/ffrivera0/reloc3r), [ROSEFusion](https://github.com/jzhzhang/ROSEFusion), and many other inspiring works in the community.


This is an open-source version of PROFusion with a reorganized architecture and rewritten functions for easier use.
While it may not reproduce the original results exactly, the output is nearly identical. 
The code has been tested on several devices with CUDA 12.4, including 4090, A800, and H800. 

