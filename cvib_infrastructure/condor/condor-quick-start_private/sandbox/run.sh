#!/bin/bash
echo $@

# change path to your username
cd /cvib2/apps/personal/{your-username}/sandbox
CUDA_VISIBLE_DEVICES=0 python mnist_ex.py
