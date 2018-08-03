#!/bin/sh

PATH_TO_MODEL_CHECKPOINT="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/train/model.ckpt-59698"
PIPELINE_CONFIG_PATH="/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/faster_rcnn_resnet50.config"
OUTPUT_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/inference_model"

python src/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${PATH_TO_MODEL_CHECKPOINT} \
    --output_directory=${OUTPUT_DIR}
