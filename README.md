# PyTorch implementation of Faster R-CNN over COCO dataset

### Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Run Inference](#run-inference)
* [Train Model](#train-model)
* [Inference Result](#inference-result)
* [Conclusion](#conclusion)

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

Prerequisite are Python3 and pip already been installed

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
Red color for the person detection and Green for the car detection here some of the result generated from this model bellow:


![image_output5](https://user-images.githubusercontent.com/30790932/133676099-c8be67e5-b694-4107-80e8-e9c450d8c63e.jpg)


![image_output3](https://user-images.githubusercontent.com/30790932/133676298-a2ab76e5-f403-466a-bed7-c287c2c23960.jpg)


![image_output1](https://user-images.githubusercontent.com/30790932/133676525-4741c84a-d149-41d7-8cb9-8a0df4680f1f.jpg)


![image_output4](https://user-images.githubusercontent.com/30790932/133676534-63e0bad0-782a-4b7d-9434-1a1b9decb98a.jpg)


![image_output](https://user-images.githubusercontent.com/30790932/133676539-2d4cc7d9-48f1-49f1-8f1f-d54daaaebd6b.jpg)


## Conclusion

For this particular approach with faster RCNN we are getting a good result but the model I have added in the repo is just trained with the 100 epochs so for the better result or more accurate result we can fine tune the model with more dataset or also we could update the backbone of the model, training with more epochs and with other hyperparameters as well.

 
