#!/bin/sh

INPUT_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/test-images"
OUTPUT_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/test-images/output"
FROZEN_GRAPH="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/graph/frozen_inference_graph.pb"
LABEL_MAP="/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/labelmap.pbtxt"
NUM_OUTPUT_CLASSES=6




python src/inference.py --input_dir=${INPUT_DIR} \
                        --output_dir=${OUTPUT_DIR} \
                        --frozen_graph=${FROZEN_GRAPH} \
                        --label_map=${LABEL_MAP} \
                        --num_output_classes=${NUM_OUTPUT_CLASSES}