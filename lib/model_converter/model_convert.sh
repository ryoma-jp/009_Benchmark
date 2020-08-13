#!/bin/bash

MODEL_DIR='../../model_000'
DATASET_DIR='../../../dataset/CIFAR-10/python/cifar-10-batches-py'
OUTPUT_DIR='./output'

python3 model_converter.py --saved_model_dir ${MODEL_DIR} --saved_model_prefix model.ckpt --node_name_yaml node_name.yaml --output_dir ${OUTPUT_DIR} --dataset_dir ${DATASET_DIR}  --dataset_type cifar10


