## Tensorflow Object Detection (with simultaneous validation on single gpu & added validation metrics)


### Requirements:
* tensorflow-gpu (1.9.0)
* CUDA-9.0 and cudnn-7.1
* virtual environment
* Object Detection API pull

### Setup:  
  Create a virtual environment (Not to mess up with your existing tensorflow installations & projects.)

Create the environment with ```virtualenv --system-site-packages tensorflow-1.9.0```. Then, activate the virtualenv

```bash
$ source path-to/tensorflow-1.9.0/bin/activate
```

and when you install things use ```pip install --ignore-installed or pip install -I ```. e.g. (```pip install -I tensorflow-gpu```), that way pip will install what you've requested locally even though a system-wide version exists. Your python interpreter will look first in the virtualenv's package directory, so those packages should shadow the global ones.
___

**Follow the below described instructions from tensorflow's object detection library (latest).**

  * <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'>Object Detection API Installation</a><br>

To **build and install** Object Detection API, 

```bash
#from models/research/
$ python setup.py build install

#from models/research/slim
$ python setup.py build install
```

### Directory Structure:
You may change the working directory structure as per your requirements.
```
.
├── data
│   ├── checkpoint (pretrained model checkpoint)
│   ├── faster_rcnn_resnet50.config (config file)
│   ├── labelmap.pbtxt
│   ├── test.record
│   └── train.record
├── export_inference_graph.sh
├── infer.sh
├── model
│   ├── inference_model
│   │   └── frozen_graph.pb file of our trained model
│   ├── train
│   │   └── train_checkpoints & summary (training dir)
│   └── val
│       └── validation_summary (validation dir)
├── src
│   ├── eval.py
│   ├── evaluator.py
│   ├── export_inference_graph.py
│   ├── inference.py
│   ├── __init__.py
│   ├── od_segmentation.py
│   ├── trainer.py
│   ├── trainer.pyc
│   ├── trainer_test.py
│   └── train.py
|── train_eval_gpu.sh

```

### Usage:

 * For *Simultaneous Validation* (training and validation on the same gpu) we have to set a gpu fraction for the training process.
 * We also have to set summary and checkpoint interval secs. (After how many seconds summary should be save)
 * As all the checkpoints will be saved, please ensure to have enough space on the disk.

#### Config File:
All the model parameters should be specified here!
```
train_input_reader {
  label_map_path: "/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/labelmap.pbtxt"
  tf_record_input_reader {
    input_path: "/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/train.record"
  }
}

eval_config {
  num_examples: 67 #number of examples to be evaluated
  #wait for 10 seconds before evaluating again (can be increased)
  eval_interval_secs: 10
  metrics_set: 'open_images_V2_detection_metrics' #different metrices can be saved
  use_moving_averages: false
}

eval_input_reader {
  label_map_path: "/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/labelmap.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/test.record"
  }
}
```

#### Training and Validation:
Source code are in ```src/``` folder.

**nohup** will send the processes to the background.

To run, from working dir:
```
$ ./train_eval_gpu.sh
```

```bash
#!/bin/sh

OBJECT_DETECTION_FOLDER="/home/yoda/Desktop/Joydeep/models/research/object_detection"
TRAIN_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/train"
VAL_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/val"
PIPELINE_CONFIG_PATH="/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/faster_rcnn_resnet50.config"

echo "Training Job..."

################ TRAINING ####################
nohup python src/train.py \
	--logtostderr \
	--train_dir=${TRAIN_DIR} \
	--gpu_fraction=0.7 \
	--save_summaries_secs=300 \
	--save_interval_secs=300 \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} > trainlog.out &


echo "Evaluation on GPU"
########### Simultaneously Evaluate on same GPU ############

nohup python src/eval.py \
	--logtostderr \
	--checkpoint_dir=${TRAIN_DIR} \
	--eval_dir=${VAL_DIR} \
	--pipeline_config_path=${PIPELINE_CONFIG_PATH} > evallog.out &

echo "Tensorboard http://localhost:6006"

nohup tensorboard --logdir model/ > tensorboard.out &

```
P.S. Be careful in choosing validation data.


#### Stop Training & validation Process
```bash
nvidia-smi

kill -9 pid_1 pid_2
```
pid_1 and pid_2 python-tensorflow process ids.

#### Export Inference Model:

Choose the best validation loss/performance checkpoint from training directory.

```bash
#!/bin/sh

PATH_TO_MODEL_CHECKPOINT="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/train/model.ckpt-59698"
PIPELINE_CONFIG_PATH="/home/yoda/Desktop/Joydeep/models/research/cards-detection/data/faster_rcnn_resnet50.config"
OUTPUT_DIR="/home/yoda/Desktop/Joydeep/models/research/cards-detection/model/inference_model"


python src/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${PATH_TO_MODEL_CHECKPOINT} \
    --output_directory=${OUTPUT_DIR}

```

The frozen graph with weights will be saved on OUTPUT_DIR.

#### Inference

* Inference can be done on folder by two scripts.
	* `inference.py`
	* `od_segmentation.py`

```bash
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
``` 
`od_segmentation.py` can also be used for inference in batches.

#### Run only evaluation on a trained checkpoint
Just run the `eval.py` script on specified trained checkpoint. It will evaluate the latest checkpoint.

___
### Tensorflow Object Detection Directory Modification (Modified Metrics):

Make changes in the following files of the object detection folder.


```
# models/research
├── object_detection
│   ├── eval_util.py
│   ├── model_lib.py
│   ├── model_main.py
│   └── utils
│       ├── metrics.py
│       ├── object_detection_evaluation.py
│       └── per_image_evaluation.py
```

* The modified method for F1 score and per category stats are written in ```object_detection_evaluation.py```  &  ```metrics.py```
* If you wish to add more evaluation metric you can add it in these files.
* After that, again do **build & install** like before to get these new features on tensorboard.

___
### Useful Definitions

#### Corloc (Correct Localization):
The Corloc is to compute the number of the right boxes you have been detected. The formulation is TP/TP+FP. The rightmost bounding box is neither TP or FP but FN. Thus if there are only 3 detections(TP+FP) with 2 positive(TP). Corloc=2/3=66%

Corloc is only about how many of the boxes your network has detected are actually right at a fixed IOU.

#### mAP
 It is the average of the maximum precisions at different recall values.

See this for more **info** [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) .


#### Model Zoo
[CoCo Trained Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)


##### this Readme will be updated periodically as new features are added!
Thank you!