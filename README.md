<div align="center">
<h1>360DVO: Deep Visual Odometry for Monocular 360-Degree Camera</h1>

<a href="https://ieeexplore.ieee.org/document/11358682">
  <img src="https://img.shields.io/badge/Paper-blue" alt="Paper">
</a>
<a href="https://arxiv.org/abs/2601.02309"><img src="https://img.shields.io/badge/arXiv-2601.02309-b31b1b" alt="arXiv"></a>
<a href="https://chris1004336379.github.io/360DVO-homepage/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/datasets/chris1004336379/360DVO"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue'></a>


**The Hong Kong University of Science and Technology**

**IEEE Robotics and Automation Letters (RA-L), 2026.**

[Xiaopeng Guo](https://scholar.google.com/citations?hl=en&user=Bi0mgawAAAAJ&inst=1381320739207392350), [Yinzhe Xu](https://scholar.google.com/citations?user=w7-kROsAAAAJ&hl=zh-TW&inst=1381320739207392350), 
[Huajian Huang](https://huajianup.github.io/), [Sai-Kit Yeung](https://saikit.org/index.html)

</div>

## Getting Started

### Installation

1. Clone 360DVO.
```bash
git clone https://github.com/chris1004336379/360DVO.git
cd 360DVO
```

2. Create the environment.
```bash
conda env create -f environment.yml
conda activate 360dvo
```

3. Install the 360DVO package
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install
pip install .
```

4. [Download weights](https://drive.google.com/file/d/17x0V5DnXye5f1qlWLK80yRG6luInPmid/view?usp=sharing)

### Demo
360DVO can be run on any 360 video or image directory with a single command. We use Viser to visualize the reconstructions in real-time.
```bash
python demo.py \
    --imagedir=<path to image directory or video> \
    --viz # enable visualization
    --plot # save trajectory plot
    --save_ply # save point cloud as a .ply file
    --save_trajectory # save the predicted trajectory as .txt in TUM format
```

### Training
comming soon

## Acknowledgements
Our code is based on [DPVO](https://github.com/princeton-vl/DPVO.git), [SphereNet](https://github.com/mty1203/spherenet.git), [Viser](https://github.com/nerfstudio-project/viser.git). We thank the authors for their excellent work!

## Citation

If you find our work useful, please cite:

```bibtex
@ARTICLE{11358682,
  author={Guo, Xiaopeng and Xu, Yinzhe and Huang, Huajian and Yeung, Sai-Kit},
  journal={IEEE Robotics and Automation Letters}, 
  title={360DVO: Deep Visual Odometry for Monocular 360-Degree Camera}, 
  year={2026},
  volume={11},
  number={3},
  pages={3079-3086},
  keywords={Feature extraction;Cameras;Nonlinear distortion;Convolution;Kernel;Bundle adjustment;Visual odometry;Accuracy;Benchmark testing;Robustness;Visual odometry;omnidirectional vision},
  doi={10.1109/LRA.2026.3655280}}
```
