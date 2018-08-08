#!/bin/sh
OBJECT_DETECTION_FOLDER="/home/yoda/Desktop/Joydeep/models/research/object_detection"

# echo "Exporting Slim Folder"

# export PYTHONPATH=$PYTHONPATH:${OBJECT_DETECTION_FOLDER}/slim


# Install Slim

TRAIN_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/train"
VAL_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/val"

PIPELINE_CONFIG_PATH="/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/faster_rcnn_resnet50.config"



echo "Training Job..."


nvidia-smi

####################### TRAINING ################################
nohup python src/train.py \
	--logtostderr \
	--train_dir=${TRAIN_DIR} \
	--gpu_fraction=0.65 \
	--save_summaries_secs=10 \
	--save_interval_secs=10 \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} > trainlog.out &


sleep 2m

echo "Evaluation on GPU"


##################### Simultaneously Evaluate on same GPU ######################

nohup python src/eval.py \
	--logtostderr \
	--checkpoint_dir=${TRAIN_DIR} \
	--eval_dir=${VAL_DIR} \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} > evallog.out &

echo "Tensorboard http://localhost:6006"

nohup tensorboard --logdir model/ > tensorboard.out &







