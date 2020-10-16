#! /bin/bash

# --- run mAP calculation ---
python3 tf_object_detection.py --dataset_type coco2014 --dataset_dir ${DATASET_DIR} --model ${MODEL_FILE} --output_dir "output_coco2014"



