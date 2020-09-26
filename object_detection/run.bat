
@echo off

rem --- coco2014 dataset ---
set DATASET_DIR=E:\work\wsl\ubuntu2004\MachineLearning\dataset\coco2014
set MODEL_FILE=model\ssdlite_mobilenet_v2_coco_2018_05_09\frozen_inference_graph.pb
set OUTPUT_DIR=output_coco2014

python tf_object_detection.py --dataset_type coco2014 --dataset_dir %DATASET_DIR% --model %MODEL_FILE% --output_dir %OUTPUT_DIR%



