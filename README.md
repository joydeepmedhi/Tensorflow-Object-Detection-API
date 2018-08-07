## Tensorflow Object Detection (with simultaneous validation on single gpu & added validation metrics)


### Requirements
* tensorflow-gpu (1.9.0)
* CUDA-9.0 and cudnn-7.1
* virtual environment

### Table of contents
### Setup:  
  Create a virtual environment (Not to mess up with your existing tensorflow installations & projects.)

Create the environment with ```virtualenv --system-site-packages tensorflow-1.9.0```. Then, activate the virtualenv

```$ source path-to/tensorflow-1.9.0/bin/activate```

and when you install things use ```pip install --ignore-installed or pip install -I ```. e.g. (```pip install -I tensorflow-gpu```), that way pip will install what you've requested locally even though a system-wide version exists. Your python interpreter will look first in the virtualenv's package directory, so those packages should shadow the global ones.
___

Follow the below described instructions from tensorflow's object detection library (latest).

  * <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'>Object Detection API Installation</a><br>

To **build and install** Object Detection API, 

```python
#from models/research/
$    python setup.py build install

#from models/research/slim
$   python setup.py build install
```






<!-- ##### Quick Start:

  * <a href='object_detection_tutorial.ipynb'>
      Quick Start: Jupyter notebook for off-the-shelf inference</a><br>
  * <a href="g3doc/running_pets.md">Quick Start: Training a pet detector</a><br> -->
  
#### 