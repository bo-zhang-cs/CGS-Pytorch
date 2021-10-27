# CGS-Pytorch
This is an unofficial PyTorch implementation of [Composing Good Shots by Exploiting Mutual Relations](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Composing_Good_Shots_by_Exploiting_Mutual_Relations_CVPR_2020_paper.html).

# Results

## GAICD
| #Metric | SRCC↑ | Acc5↑ | Acc10↑ |
|:--:|:--:|:--:|:--:|
| Paper   | 0.795 | 59.7  | 77.8   |
| This code (best SRCC) |  0.790  |  57.8  | 74.6 |
| This code (best Acc)  |  0.779  |  59.5  | 77.3 |   

I set the probability of mixing graph as 0.3 druing training, and scale the elements of adjacency matrix by the number of crops to produce more stable score prediction. 

## HCDB
| #Metric | IoU↑ | BDE↓ |
|:--:|:--:|:--:|
| Paper   | 0.836 | 0.039 |
| This code | 0.811  |  0.044  |

# Datasets Preparation
+ GAICD [[link]](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping)
+ HCDB (FLMS) [[Download Images]](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_image/image.tar) [[Download Annotation]](http://fangchen.org/proj_page/FLMS_mm14/data/radomir500_gt/release_data.tar)

Download&Unzip these datasets, palce them like this:
  
    DATASET_FOLDER
      ├── GAICD
      │   └── images
      │   │    ├── image1.jpg
      │   │    └── image2.jpg
      │   └── annotations 
      │           ├── image1.txt
      │           └── image2.txt
      └── FLMS
          └── image
          │    ├── image1.jpg
          │    └── image2.jpg
          └── 500_image_dataset.mat 

# Usage
```bash
  # clone this repository
  git clone https://github.com/bo-zhang-cs/CGS-Pytorch.git
  cd CGS-Pytorch
  ```
Change the default dataset folder in ``config.py`` and you can check the paths by running ``cropping_dataset.py``.

## Install RoIAlign and RoDAlign

The source code of RoI&RoDAlign is from [[here]](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch) compatible with PyTorch 1.0 or later.
If you use Pytorch 0.4.1, please refer to [[official implementation]](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch).

1. Change the **CUDA_HOME** and **-arch=sm_86** in ``roi_align/make.sh`` and ``rod_align/make.sh`` according to your enviroment, respectively.
2. If you run this code in linux envoriment, make sure these bash files (``make_all.sh, roi_align/make.sh, rod_align/make.sh``) are Unix text file format by runing ``:set ff=unix`` in VIM.
3. ``cd CGS-Pytorch && sudo bash make_all.sh`` to build and install the packages. 

## Test

Download pretrained models (~75MB, ZIP format file) from [[Google Drive]](https://drive.google.com/file/d/1CmMBcQdOc22Qnyle0xJlbO6BC8tdViWN/view?usp=sharing) and unzip to the folder ``CGS-Pytorch/pretrained_model``.
```
python test.py
```
This will produce a folder ``results`` where you can find the predicted best crops.

## Train
```
python train.py
```
Track training process:
```
tensorboard --logdir=./experiments --bind_all
```
The model performance for each epoch is also recorded in *.csv* file under the produced folder *./experiments*.

# Citation
```
@inproceedings{li2020composing,
  title={Composing good shots by exploiting mutual relations},
  author={Li, Debang and Zhang, Junge and Huang, Kaiqi and Yang, Ming-Hsuan},
  booktitle={CVPR},
  year={2020}
}
@inproceedings{zeng2019reliable,
  title={Reliable and efficient image cropping: A grid anchor based approach},
  author={Zeng, Hui and Li, Lida and Cao, Zisheng and Zhang, Lei},
  booktitle={CVPR},
  year={2019}
}
```

## More references about image cropping 
[Awesome Image Aesthetic Assessment and Cropping](https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping)

## Acknowledgments
Thanks to [[GAIC]](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch) and [[GAIC-Pytorch1.0+]](https://github.com/lld533/Grid-Anchor-based-Image-Cropping-Pytorch).
