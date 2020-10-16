
@echo off

rem --- coco2014 dataset ---
set DATASET_DIR=E:\work\wsl\ubuntu2004\MachineLearning\dataset\coco2014
goto ssdlite_mobilenet_v2_coco_2018_05_09

rem --- http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz ---
:ssdlite_mobilenet_v2_coco_2018_05_09
set MODEL_FILE=model\ssdlite_mobilenet_v2_coco_2018_05_09\frozen_inference_graph.pb
set OUTPUT_DIR=output_coco2014-ssdlite_mobilenet_v2_coco_2018_05_09
goto run

rem --- http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz ---
:ssd_mobilenet_v2_coco_2018_03_29
set MODEL_FILE=model\ssd_mobilenet_v2_coco_2018_03_29\frozen_inference_graph.pb
set OUTPUT_DIR=output_coco2014-ssd_mobilenet_v2_coco_2018_03_29
goto run

rem --- run ---
:run
python tf_object_detection.py --dataset_type coco2014 --dataset_dir %DATASET_DIR% --model %MODEL_FILE% --output_dir %OUTPUT_DIR%



