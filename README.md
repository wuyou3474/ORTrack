# ORTrack
The official implementation for the **CVPR 2025** paper [ [**Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking**](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Learning_Occlusion-Robust_Vision_Transformers_for_Real-Time_UAV_Tracking_CVPR_2025_paper.pdf) ]

[Models & Raw Results](https://pan.baidu.com/s/1Ym4J1b5RzyqvcEgNFyeX1A?pwd=cvpr) Baidu Driver [Models & Raw Results](https://drive.google.com/drive/folders/1pXq5HHezldjOyFCj6DUrXl5H8r4KiGv3?usp=sharing) Google Driver

##  Methodology

<p align="center">
  <img width="85%" src="assets/ORTrack.png" alt="ORTrack"/>
</p>


## Usage
### Installation
Create and activate a conda environment:
```
conda create -n ORTrack python=3.8
conda activate ORTrack
```

Install the required packages:
```
pip install -r requirements.txt
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST         
   ```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_ORTrack>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```

### Training
Download pre-trained [DeiT-Tiny weights](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth), [Eva02-Tiny weights](https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/pt/eva02_Ti_pt_in21k_p14.pt) , and [ViT-Tiny weights](https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz)  and put it under `$USER_ROOT$/.cache/torch/hub/checkpoints/. 
```
# Training ORTrack-DeiT
python tracking/train.py --script ortrack --config deit_tiny_patch16_224  --save_dir ./output --mode single

# Training ORTrack-D-DeiT
# You need to download the model weight of ORTrack-DeiT and place them under the directory $PROJECT_ROOT$/teacher_model/deit_tiny_patch16_224.
python tracking/train.py --script ortrack --config deit_tiny_distilled_patch16_224  --save_dir ./output --mode single
```


### Testing
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1pXq5HHezldjOyFCj6DUrXl5H8r4KiGv3?usp=sharing) or [BaiduNetDisk](https://pan.baidu.com/s/1Ym4J1b5RzyqvcEgNFyeX1A?pwd=cvpr)

Put the downloaded weights on `<PATH_of_ORTrack>/output/checkpoints/train/ortrack/deit_tiny_patch16_224`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

 Testing examples:
- VisDrone2018 or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py ortrack deit_tiny_patch16_224 --dataset visdrone2018 --threads 4 --num_gpus 1
python tracking/analysis_results.py # need to modify tracker configs and names
```
- BioDrone
```
python tracking/test.py ortrack deit_tiny_patch16_224 --dataset biodrone --threads 4 --num_gpus 1
```


### Test FLOPs, and Params.

```
# Profiling ORTrack-DeiT
python tracking/profile_model.py --script ortrack --config deit_tiny_patch16_224
```


## Acknowledgment
* This repo is based on [OSTrack](https://github.com/botaoye/OSTrack) and [PyTracking](https://github.com/visionml/pytracking) library which are excellent works and help us to quickly implement our ideas.

* We use the implementation of the DeiT, Eva02, and ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo. 


## Citation
If our work is useful for your research, please consider citing:
```Bibtex
@inproceedings{wu2025ortrack,
  title={Learning Occlusion-Robust Vision Transformers for Real-Time UAV Tracking},
  author={Wu, You and Wang, Xucheng and Yang, Xiangyang and Liu, Mengyuan and Zeng, Dan and Ye, Hengzhou and Li, Shuiwang},
  booktitle={CVPR},
  year={2025}
}
```

