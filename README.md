# Face-mask-Detection-using-Jetson-Nano-2GB
# Aim And Objectives

# Aim

To create a Face-mask detection system which will detect Human face and then check if mask is worn or not.

# Objectives

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine learning.

• To show on the optical viewfinder of the camera module whether a person is wearing a mask or not.

# Abstract

• A person’s face is classified whether a helmet is worn or not and is detected by the live feed from the system’s camera.

• We have completed this project on jetson nano which is a very small computational device.

• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.

• The continuous spread of the virus forced governments of various countries to put lockdown for several months. It has been observed that wearing a face mask can actually prevent the transmission of this deadly virus.

• In the future, we have to use a face mask as a preventive measure for any such viruses. This project highlights the importance of YOLOv5 especially object detection.

# Introduction

• The spread of COVID-19 is increasingly worrying for everyone in the world. This virus can be affected from human to human through the droplets and airborne.

• According to the instruction from WHO, to reduce the spread of COVID-19, every people need to wear face mask, do social distancing, evade the crowd area and also always maintain the immune system.

• Therefore, to protect each other, every person should wear the mask properly when they are in outdoor.

• However, most of selfish people won't wear the face mask properly with so many reasons.

• To overcome this situation, a robust face mask detection needs to be developed. In order to detect a face mask, the object detection algorithm can be implemented.

# Literature Review

• The face mask detection model is very useful for public places like hospitals, airports, offices where a huge number of people
travel from one place to another.

• In hospitals, we can embed this model in pre-installed CCTV cameras. If the workers of the hospitals are found without mask alarm will ring and the higher authorities of the hospital can take necessary actions against the worker.

• In airports, the entrance and exit gate of the airport should have this model.

• The System is prepared to recognize precisely whether an individual is wearing a mask or not. At the point when the calculation recognizes an individual without a mask, caution ought to be produced to alarm the individuals around or the concerned specialists close by, so fundamental activities can be taken against such violators.

• Not only for Covid19 pandemic, any place and at whatever point
facemask is commanded to relieve any air-borne illnesses, passage, what's more, leave access frameworks can be incorporated with such innovation to help in diminishing the spread of infection.

•  The cameras are used to capture images from public places; then these images are feed into a system that identifies if any person without face mask appears in the image. If any person without a face mask is detected then this information is sent to the proper authority to take necessary actions.

# Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

# Jetson Nano 2GB

https://user-images.githubusercontent.com/112484744/225980164-d5b8b51d-64a3-462c-bd4f-89164db73c7e.mp4

# Proposed System

1] Study basics of machine learning and image recognition.

2] Start with implementation



```bash
• Front-end development
• Back-end development
```


3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether a person is wearing a mask or not.

4] Use datasets to interpret the object and suggest whether the person on the camera’s viewfinder is wearing a mask or not.

# Methodology

#### The Face-mask detection system is a program that focuses on implementing real time face-mask detection.

#### It is a prototype of a new product that comprises of the main module:

#### Face-mask detection and then showing on viewfinder whether the person is wearing a mask or not.

#### Face-mask Detection Module

## This Module is divided into two parts:

#### 1] Face detection

• Ability to detect the location of a person’s face in any input image or frame. The output is the bounding box coordinates on the detected face of a person.

• For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.

• This Datasets identifies person’s face in a Bitmap graphic object and returns the bounding box image with annotation of mask or no mask present in each image.

#### 2] Mask Detection

• Recognition of the face and whether mask is worn or not.

• Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.

• There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.

• YOLOv5 was used to train and test our model for whether the mask was worn or not. We trained it for 149 epochs and achieved an accuracy of approximately 92%.

# Installation


#### Initial Configuration

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```

#### Create Swap

```bash
udo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
# make entry in fstab file
/swapfile1	swap	swap	defaults	0 0
```

#### Cuda env in bashrc

```bash
vim ~/.bashrc

# add this lines
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

#### Update & Upgrade

```bash
sudo apt-get update
sudo apt-get upgrade
```
#### nstall some required Packages

```bash
sudo apt install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo apt-get install libopenblas-base libopenmpi-dev
```
#### Install Torch

```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" 
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
#### Install Torchvision

```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
#### Clone Yolov5

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
#### Download weights and Test Yolov5 Installation on USB webcam

```bash
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt  --source 0
```
# Face-mask Dataset Training

### We used Google Colab And Roboflow

#### train your model on colab and download the weights and pass them into yolov5 folder.

# Running Face-mask Detection Model

source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0
```
# Demo

https://user-images.githubusercontent.com/112484744/226101557-29a356f5-5b88-4fa2-9641-09e2c600f8ef.mp4




# Advantages







• Public places like Bus stand, Air ports and Railway stations

• Offices and Education institutes

#### Benefits

• Cost effective

• Curb Covid-19 pandemic

• Life Saving

# Conclusion 

• Efficient Image capturing

• Efficient Dataset training through yolov5

• Successful face mask detection

• Maintaining alert status 

# Future scope 

• In the future, physical distance integration could be introduced as a feature, or coughing and sneezing detection could be added.

• Apart from detecting the face mask, it will also compute the distances among each individual and see any possibility of coughing or sneezing. If the mask is not worn properly, a third class can be introduced that labels the image as ‘improper mask'.

• In addition, researchers could propose a better optimiser, improved parameter configuration, and the use of adaptive models.



# Reference

1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

# Article

https://doi.org/10.1109/JSEN.2021.3061178

https://doi.org/10.1109/LSP.2020.3032277
