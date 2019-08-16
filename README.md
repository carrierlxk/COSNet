# COSNet
Code for CVPR 2019 paper: See More, Know More: Unsupervised Video Object Segmentation with
Co-Attention Siamese Networks.[Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_See_More_Know_More_Unsupervised_Video_Object_Segmentation_With_Co-Attention_CVPR_2019_paper.pdf)

##

![](../master/framework.png)

###The pre-trained model and testing code:

### Quick Start

1. Install pytorch (version:1.0.1).

2. Download the pretrained model. Run 'test_coattention_conf.py' and change the davis dataset path, pretrainde model path and result path.

3. Run command: python test_coattention_conf.py --dataset davis --gpus 0

4.Post CRF processing code: https://github.com/lucasb-eyer/pydensecrf

The pretrained weight can be download from [GoogleDrive](https://drive.google.com/open?id=14ya3ZkneeHsegCgDrvkuFtGoAfVRgErz) or [BaiduPan](https://pan.baidu.com/s/16oFzRmn4Meuq83fCYr4boQ), pass code: xwup.

The segmentation results on DAVIS, FBMS and Youtube-objects can be download from [GoogleDrive](https://drive.google.com/open?id=1DuNYAXPoCRyFmV7fGOuImIqJsFjEgKT4),[BaiduPan](https://pan.baidu.com/s/1528vUQLH98qV93dsH-wqhw), pass code: nzbq.

### Citation
If you find the code and dataset useful in your research, please consider citing:

@InProceedings{Lu_2019_CVPR,

author = {Lu, Xiankai and Wang, Wenguan and Ma, Chao and Shen, Jianbing and Shao, Ling and Porikli, Fatih},

title = {See More, Know More: Unsupervised Video Object Segmentation With Co-Attention Siamese Networks},

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},

year = {2019}
}
