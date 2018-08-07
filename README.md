## Tensorflow Object Detection (with simultaneous validation on single gpu & added validation metrics)


### Requirements:
* tensorflow-gpu (1.9.0)
* CUDA-9.0 and cudnn-7.1
* virtual environment

### Setup:  
  Create a virtual environment (Not to mess up with your existing tensorflow installations & projects.)

Create the environment with ```virtualenv --system-site-packages tensorflow-1.9.0```. Then, activate the virtualenv

```bash
$ source path-to/tensorflow-1.9.0/bin/activate
```

and when you install things use ```pip install --ignore-installed or pip install -I ```. e.g. (```pip install -I tensorflow-gpu```), that way pip will install what you've requested locally even though a system-wide version exists. Your python interpreter will look first in the virtualenv's package directory, so those packages should shadow the global ones.
___

Follow the below described instructions from tensorflow's object detection library (latest).

  * <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'>Object Detection API Installation</a><br>

To **build and install** Object Detection API, 

```python
#from models/research/
$ python setup.py build install

#from models/research/slim
$ python setup.py build install
```

#### Directory Structure:
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

### Usage

 * For *Simultaneous Validation* (training and validation on the same gpu) we have to set a gpu fraction for the training process.
 * We also have to set summary and checkpoint interval secs. (After how many seconds summary should be save)
 * As all the checkpoints will be saved, please ensure to have enough space on the disk.

#### Training and Validation
Source code are in ```src/``` folder.

from working dir:  ```./train_eval_gpu.sh```

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

### Tensorflow Object Detection Directory Modification(Modified Metrics):

Make changes in the following files of the object detection


```
├── object_detection
│   ├── eval_util.py
│   ├── model_lib.py
│   ├── model_main.py
│   └── utils
│       ├── metrics.py
│       ├── object_detection_evaluation.py
│       └── per_image_evaluation.py
```


The modified method for F1 score and per category stats are written in ```object_detection_evaluation.py & metrics.py``` 