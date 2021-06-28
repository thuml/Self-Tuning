# Self-Tuning for Data-Efficient Deep Learning

This repository contains the implementation code for paper: <br>
__Self-Tuning for Data-Efficient Deep Learning__ <br>
[Ximei Wang](https://wxm17.github.io/), [Jinghan Gao](https://github.com/getterk96), [Mingsheng Long](http://ise.thss.tsinghua.edu.cn/~mlong/), [Jianmin Wang](http://ise.thss.tsinghua.edu.cn/~wangjianmin/)<br>
_38th International Conference on Machine Learning (ICML 2021)_ <br>
[[Project Page](http://github.com/thuml/Self-Tuning)] [[Paper](https://arxiv.org/abs/2102.12903)] [[Video](https://recorder-v3.slideslive.com/?share=40334&s=f7988e61-bece-4a7a-a6ba-3e1a2b49b37b)] [Blog]

<p align="center">
    <img src="Self-Tuning.png" width="500"> <br>
<b>Data-Efficient Deep Learning (DEDL)</b> aims to mitigate the requirement of labeled data by unifying the exploration of labeled and unlabeled data and the transfer of a pre-trained model.
</p>


## Beyond Transfer Learning and Semi-supervised Learning: Brief Introduction for DEDL
Mitigating the requirement for labeled data is a vital issue in deep learning community. However, _common practices_ of TL and SSL only focus on either the pre-trained model or unlabeled data. This paper unleashes the power of both worlds by proposing _a new setup_ named data-efficient deep learning, aims to mitigate the requirement of labeled data by unifying the exploration of labeled and unlabeled data and the transfer of pre-trained model. 

To address the challenge of confirmation bias in self-training, a general _Pseudo Group Contrast_ mechanism is devised to mitigate the reliance on pseudo-labels and boost the tolerance to false labels. To tackle the model shift problem, we unify the exploration of labeled and unlabeled data and the transfer of a pre-trained model, with a shared key queue beyond just 'parallel training'. 
Comprehensive experiments demonstrate that Self-Tuning outperforms its SSL and TL counterparts on five tasks by sharp margins, _e.g._, it doubles the accuracy of fine-tuning on Stanford-Cars provided with 15% labels. 


## Dependencies
* python3.6
* torch == 1.3.1 (with suitable CUDA and CuDNN version)
* torchvision == 0.4.2
* tensorboardX
* numpy
* argparse

## Datasets
| Dataset | Download Link |
| -- | -- |
| CUB-200-2011 | http://www.vision.caltech.edu/visipedia/CUB-200-2011.html |
| Stanford Cars | http://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| FGVC Aircraft | http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |

## Quick Start
- The running commands for several datasets are shown below. Please refer to ``run.sh`` for commands for datasets with other label ratios.
```
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./cifar100 --batch_size 24 --logdir vis/ --gpu_id 3 --queue_size 16 --backbone efficientnet-b2 --num_labeled 10000 --expand_label --pretrained --projector_dim 1024

```




## Updates
- [06/2021] A five minute video is released to briefly introduce the main idea of Self-Tuning. We have released the code and models. You can find all reproduced checkpoints via [this link](https://cloud.tsinghua.edu.cn/d/4e8fb444c4634e76ab0a/).
- [05/2021] Paper accepted to ICML 2021 as a __Short Talk__. 
- [02/2021] [arXiv version](https://arxiv.org/abs/2102.12903) posted. Please stay tuned for updates.


## Citation
If you find this code or idea useful, please cite our work:
```bib
@inproceedings{wang2021selftuning,
  title={Self-Tuning for Data-Efficient Deep Learning},
  author={Wang, Ximei and Gao, Jinghan and Long, Mingsheng and Wang, Jianmin},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2021}
}
```


## Contact
If you have any questions, feel free to contact us through email (wxm17@mails.tsinghua.edu.cn) or Github issues. Enjoy!
