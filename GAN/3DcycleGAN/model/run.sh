#!/bin/bash
echo $@

cd /cvib2/apps/personal/swelland/sandbox/gan/cyclegan-ct-abdomen/model-0
CUDA_VISIBLE_DEVICES=0 python cyclegan_resnet.py
