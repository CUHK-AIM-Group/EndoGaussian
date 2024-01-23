# EndoGaussian: Gaussian Splatting for Deformable Surgical Scene Reconstruction

## arXiv Preprint

### [Project Page]()| [arXiv Paper]()


[Yifan Liu](https://guanjunwu.github.io/)<sup>1*</sup>, [Chenxin Li](https://github.com/taoranyi)<sup>1*</sup>,
[Chen Yang](https://jaminfong.cn/)<sup>2</sup>, [Yixuan Yuan](http://lingxixie.com/)<sup>1✉</sup>

<sup>1</sup>Department of Electronic Engineering, CUHK &emsp; <sup>2</sup>Department of Electrical Engineering &emsp;

<sup>\*</sup> Equal Contributions. <sup>✉</sup> Corresponding Author. 

-------------------------------------------

---

## Environmental Setups
Please follow the [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.
```bash
git clone https://github.com/hustvl/4DGaussians
cd 4DGaussians
git submodule update --init --recursive
conda create -n Gaussians4D python=3.7 
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```
In our environment, we use pytorch=1.13.1+cu116.
## Data Preparation
**For synthetic scenes:**  
The dataset provided in [D-NeRF](https://github.com/albertpumarola/D-NeRF) is used. You can download the dataset from [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0).

**For real dynamic scenes:**  
The dataset provided in [HyperNeRF](https://github.com/google/hypernerf) is used. You can download scenes from [Hypernerf Dataset](https://github.com/google/hypernerf/releases/tag/v0.1) and organize them as [Nerfies](https://github.com/google/nerfies#datasets). Meanwhile, [Plenoptic Dataset](https://github.com/facebookresearch/Neural_3D_Video) could be downloaded from their official websites. To save the memory, you should extract the frames of each video and then organize your dataset as follows.
```
├── data
│   | dnerf 
│     ├── mutant
│     ├── standup 
│     ├── ...
│   | hypernerf
│     ├── interp
│     ├── misc
│     ├── virg
│   | dynerf
│     ├── cook_spinach
│       ├── cam00
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── 0002.png
│               ├── ...
│       ├── cam01
│           ├── images
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── cut_roasted_beef
|     ├── ...
```


## Training
For training synthetic scenes such as `bouncingballs`, run 
``` 
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
``` 
You can customize your training config through the config files.
## Rendering
Run the following script to render the images.  

```
python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py  &
```


## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```
## Scripts

There are some helpful scripts in `scripts/`, please feel free to use them.

---
## Contributions

**This project is still under development. Please feel free to raise issues or submit pull requests to contribute to our codebase.**

---
Some source code of ours is borrowed from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [k-planes](https://github.com/Giodiro/kplanes_nerfstudio),[HexPlane](https://github.com/Caoang327/HexPlane), [TiNeuVox](https://github.com/hustvl/TiNeuVox). We sincerely appreciate the excellent works of these authors.

## Acknowledgement

We would like to express our sincere gratitude to [@zhouzhenghong-gt](https://github.com/zhouzhenghong-gt/) for his revisions to our code and discussions on the content of our paper.
## Citation
Some insights about neural voxel grids and dynamic scenes reconstruction originate from [TiNeuVox](https://github.com/hustvl/TiNeuVox). If you find this repository/work helpful in your research, welcome to cite these papers and give a ⭐. 
```
@article{wu20234dgaussians,
  title={4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
  author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei Wei and Liu, Wenyu and Tian, Qi and Wang Xinggang},
  journal={arXiv preprint arXiv:2310.08528},
  year={2023}
}

@inproceedings{TiNeuVox,
  author = {Fang, Jiemin and Yi, Taoran and Wang, Xinggang and Xie, Lingxi and Zhang, Xiaopeng and Liu, Wenyu and Nie\ss{}ner, Matthias and Tian, Qi},
  title = {Fast Dynamic Radiance Fields with Time-Aware Neural Voxels},
  year = {2022},
  booktitle = {SIGGRAPH Asia 2022 Conference Papers}
}
```
