# GMFlowNet

This repository contains the official implementation for the paper:

[Global Matching with Overlapping Attention for Optical Flow Estimation](https://arxiv.org/abs/2203.11335)<br/>
CVPR 2022 <br/>
Shiyu Zhao, Long Zhao, Zhixing Zhang, Enyu Zhou, Dimitris Metaxas<br/>

## Requirements
The code has been tested with PyTorch 1.7 and Cuda 11.0. Later PyTorch may also work.
```Shell
conda create --name gmflownet
conda activate gmflownet
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install matplotlib tensorboard scipy opencv
```

## Demos
Download .zip file with pretrained models at [Google Drive](https://drive.google.com/file/d/1rVfu0j9O1M2hNsew-dVRlx9jL7c6ICru/view?usp=sharing). Unzip `pretrained_models.zip` in the root.
```Shell
unzip pretrained_models.zip
```

You can demo a trained model on a sequence of frames
```Shell
python demo.py --model gmflownet --ckpt=pretrained_models/gmflownet-things.pth --path=demo-frames
```

## Required Data
To evaluate/train RAFT, you need to download the following datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

Place all datasets in your preferred directory and symbolic link it to `./datasets` with `ln -s <your_directory> ./datasets` so that your `./datasets` folder looks like
```Shell
├── datasets
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ...
```

## Evaluation
Download the pretraind model described in [Demo](https://github.com/xiaofeng94/GMFlowNet/blob/master/README.md#demos). 
You may evaluate a pretrained model using `evaluate.py`. To get the best result,

On Sintel, evaluate the `gmflownet_mix` model as,
```Shell
python evaluate.py --model gmflownet --use_mix_attn --ckpt=pretrained_models/gmflownet_mix-things.pth --dataset=sintel
```
On KITTI, evaluate the `gmflownet` model as,
```Shell
python evaluate.py --model gmflownet --ckpt=pretrained_models/gmflownet-things.pth --dataset=kitti
```
Note: `gmflownet_mix` replaces half of heads (4 out of 8 heads) in each POLA attention of `gmflownet` with heads of [axial attentions](https://arxiv.org/abs/2003.07853) and achieves better results on Sintel.


## Training
We used the following training schedules in our paper (2 GPUs). 

- Train `gmflownet` as,
```Shell
./train_gmflownet.sh
```
- Train `gmflownet_mix` as,
```Shell
./train_gmflownet_mix.sh
```

Training logs will be written to the `./runs` which can be visualized using tensorboard as,
```Shell
tensorboard --bind_all --port 8080 --logdir ./runs
```


## Acknowledgement
The code is based on [RAFT](https://github.com/princeton-vl/RAFT) and [SwinTransformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).
We sincerely thank the authors for their great work.




