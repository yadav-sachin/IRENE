#!/bin/bash

dataset="LF-AOL-270K_10"

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python train.py configs/${dataset}/meta_clf_gen.yaml
