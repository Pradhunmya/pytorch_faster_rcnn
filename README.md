# PyTorch implementation of Faster R-CNN over COCO dataset

### Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Run Inference](#run-inference)
* [Train Model](#train-model)
* [Inference Result](#inference-result)

## Introduction  
Pytorch based implementation of faster rcnn framework for detecting the Person and car which is trained on the coco dataset.For details about faster R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

This detection framework has the following features:  
* It can be run as pure python code, and also pure based on pytorch framework, no need to build.
* It is easily trained by only running a train.py script, just set the path of the coco dataset in the config.py file.
* It can be a really detection framework. You only need to change super parameters in config file and get different models to compare different model.


## Installation
* Python 3.6 or 3.7  
* Pytorch 1.7  
* torchvision 0.10 
* numpy 1.21
* Opencv-python 4.5
* Pillow 8.3
* pycocotools 2.0
* matplotlib 3.0.2
  
```Shell
pip install -r requirements.txt
  ```


## Run Inference

Follow the below steps to run the inference:

1. First, clone this repo

```Shell
git clone https://github.com/Pradhunmya/pytorch_faster_rcnn
  ```
2. Create a virtual environment (optional)  

```Shell
pip install virtualenv
virtualenv -p python3 env
source env/bin/activate
  ```
  
3. Install the requirements from file

```Shell
cd pytorch_faster_rcnn
pip install -r requirements.txt
  ```
4. Download the Trained model and put it inside the model directory
```Shell
cd model
  ```
Downlod the model from this link: https://drive.google.com/file/d/1QyCf-ar8L1r5AOK4nRhoTTdAluKUjKC9/view?usp=sharing

5. Run the Inference Script

```Shell
python run_inference.py -i <Image PATH> -m model/model.pt
  ```
After that you will get the output display which is detecting the car and person.

## Train Model

Follow all the steps from 1 to 3 above #3 Run Inference section

4. Download the dataset and then extract inside the trainval directory

```Shell
cd trainval
  ```
Download the dataset from this link https://evp-ml-data.s3.us-east-2.amazonaws.com/ml-interview/openimages-personcar/trainval.tar.gz

Directory Structure should be after Extraction:

```
    trainval
        |-- annotations
                |--bbox-annotations.json
        |-- images
                |-- image_0000001.jpg
                |-- image_0000002.jpg
                |-- ...
        
   ```  
Please update the config.py as per the dataset path, train_data_dir and train_coco.

5. Run the training script

```Shell
python train.py
  ```

## Inference Result

![image_output4](https://user-images.githubusercontent.com/30790932/133673890-98418b94-c4e3-401c-b279-5cf65776c01c.jpg)










 
