#!/usr/bin/env bash

python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 1 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 30 --pretrained
python src/main.py  --root ./Aircraft --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 50 --pretrained

python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 15 --pretrained
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 1 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 30 --pretrained
python src/main.py  --root ./StanfordCars --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50  --label_ratio 50 --pretrained

python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 0 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 15 --pretrained
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 1 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 30 --pretrained
python src/main.py  --root ./CUB200 --batch_size 24 --logdir vis/ --gpu_id 2 --queue_size 32 --projector_dim 1024 --backbone resnet50 --label_ratio 50 --pretrained

python src/main.py  --root ./cifar100 --batch_size 20 --logdir vis/ --gpu_id 0 --queue_size 32 --backbone efficientnet-b2 --num_labeled 400 --expand_label --pretrained --projector_dim 1024
python src/main.py  --root ./cifar100 --batch_size 20 --logdir vis/ --gpu_id 1 --queue_size 32 --backbone efficientnet-b2 --num_labeled 2500 --expand_label --pretrained --projector_dim 1024
python src/main.py  --root ./cifar100 --batch_size 20 --logdir vis/ --gpu_id 3 --queue_size 32 --backbone efficientnet-b2 --num_labeled 10000 --expand_label --pretrained --projector_dim 1024
