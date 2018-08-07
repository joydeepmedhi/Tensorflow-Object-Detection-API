## Tensorflow Object Detection (with simultaneous validation on single gpu & added validation metrics)


### Requirements
* tensorflow-gpu (1.9.0)
* CUDA-9.0 and cudnn-7.1
* virtual environment

#### Table of contents
Please follow the installation guide from Official Tensorflow Object Detection API repo.

##### Setup:  
  Create a virtual environment (Not to mess up with your existing tensorflow installation.

Create the environment with ```virtualenv --system-site-packages tensorflow-1.9.0```. Then, activate the virtualenv

```$ source path-to/tensorflow-1.9.0/bin/activate```

and when you install things use ```pip install --ignore-installed or pip install -I ```. e.g. (```pip install -I tensorflow-gpu```)That way pip will install what you've requested locally even though a system-wide version exists. Your python interpreter will look first in the virtualenv's package directory, so those packages should shadow the global ones.

Follow the below described method from tensorflow's object detection library

  * <a href='g3doc/installation.md'>Object Detection API Installation</a><br>

<!-- ##### Quick Start:

  * <a href='object_detection_tutorial.ipynb'>
      Quick Start: Jupyter notebook for off-the-shelf inference</a><br>
  * <a href="g3doc/running_pets.md">Quick Start: Training a pet detector</a><br> -->
  
#### 