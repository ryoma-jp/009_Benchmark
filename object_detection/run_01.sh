#! /bin/bash

# --- set environment ---

export CUR_DIR=${PWD}
#export DATASET_DIR=/media/pi/FA80CEC380CE861B/
export DATASET_DIR=/work/MachineLearning/dataset/coco2014

# --- download dataset ---
#mkdir -p ${DATASET_DIR}
#cd ${DATASET_DIR}
#
#wget http://images.cocodataset.org/zips/val2014.zip
#unzip val2014.zip
#
#wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
#unzip annotations_trainval2014.zip
#
### bak ###
#wget http://images.cocodataset.org/zips/test2014.zip
#unzip test2014.zip
#
#wget http://images.cocodataset.org/annotations/image_info_test2014.zip
#unzip image_info_test2014.zip
#
#cd ${CUR_DIR}
#exit

# --- download model ---
if [ ! -e ./model/ssdlite_mobilenet_v2_coco_2018_05_09 ]; then
	echo "Download model: ssdlite_mobilenet_v2_coco_2018_05_09"
	mkdir -p model
	cd model
	wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
	wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssdlite_mobilenet_v2_coco.config
	tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
	cd ..
else
	echo "Skip download model"

fi
export MODEL_FILE="model/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb"

# --- download cocoapi ---
if [ ! -e ./cocoapi ]; then
	echo "Download cocoapi"
	git clone https://github.com/cocodataset/cocoapi.git

	echo
	echo "---------------------------------------------------------------"
	echo "run \"make\" under coco/PythonAPI"
	echo "if the below error is occurred, replace from \"python\" to \"python3\" in Makefile"
	echo "   pycocotools/_mask.c: No such file or directory"
	echo "---------------------------------------------------------------"
else
	echo "Skip download cocoapi"
fi

# --- download tensorflow mscoco minival ids ---
if [ ! -e ./mscoco_minival_ids.txt ]; then
	echo "Download mscoco minival ids"
	wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_minival_ids.txt
else
	echo "Skip download mscoco_minival_ids"
fi


