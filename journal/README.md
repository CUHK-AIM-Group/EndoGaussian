The journal version of EndoGaussian. Still under construction.
<!-- ![introduction](assets/teaser.png) -->

## Difference with arXiv version
1. Add dataset support for Hamlyn dataset
2. Foundation-model guided initialization to automatically segment tool masks
3. Motion-aware frame synthesis to interpolate new frames for further supervision

## 📚 Data Preparation
**EndoNeRF:**  
The dataset provided in [EndoNeRF](https://arxiv.org/abs/2206.15255) is used. You can download and process the dataset from their website (https://github.com/med-air/EndoNeRF). We use the two accessible clips including 'pulling_soft_tissues' and 'cutting_tissues_twice'.

**Hamlyn:**  
The dataset provided in [Forplane](https://github.com/Loping151/ForPlane) is used. Thanks for their efforts.

## ⚙️ Data Processing
**Mask generation:**  
We use [GroundedSAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) with text prompts "instrument, tools" to automatically get instrument masks. After obtaining the mask files, we move them into the corresponding dataset folder.

**Depth prediction:**  
For binocular inputs, we use [STTR](https://github.com/mli0603/stereo-transformer) to predict. For monocular inputs, we use [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2) to predict depth maps;  Similarly, the results are moved into the dataset folder.

**Frame interpolation**  
We interpolate fraes beforehead using the frame interpolation model [RIFE](https://github.com/hzwer/ECCV2022-RIFE), where the interpolation ratio can be defined. Here we shown the example of ratio 2.

The resulted file structure is as follows.
```
├── data
│   | endonerf 
│     ├── pulling
│     ├── cutting 
│   | hamlyn
│     ├── hamlyn_seq1
|         ├── images
|         ├── images_interp2 (predicted)
|         ├── masks (predicted)
|         ├── depth (predicted)
|     ├── hamlyn_seq2
|   | Your dataset
```

## ⏳ Training
For training scenes such as `pulling_soft_tissues`, run 
``` 
python train.py -s data/endonerf/pulling --port 6017 --expname endonerf/pulling --configs arguments/endonerf/pulling.py 
``` 
You can customize your training config through the config files.
## 🎇 Rendering & Reconstruction(optional)
Run the following script to render the images.  

```
python render.py --model_path output/endonerf/pulling  --skip_train --skip_video --configs arguments/endonerf/pulling.py
```
You can use `--skip_train`, `--skip_test`, and `--skip_video` to skip rendering images of training, testing, and video set. By default, all three sets are rendered.

Besides, we also provide point cloud reconstruction function, you can add extra arguments `--reconstruct` to activate it.

## 📏 Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --model_path output/endonerf/pulling
```

## 📜 Citation
If you find this repository/work helpful in your research, welcome to cite this paper and give a ⭐. 
```
@article{liu2025foundation,
  title={Foundation Model-guided Gaussian Splatting for 4D Reconstruction of Deformable Tissues},
  author={Liu, Yifan and Li, Chenxin and Liu, Hengyu and Yang, Chen and Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```
