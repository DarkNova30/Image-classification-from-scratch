# Cat vs Dog Image Classification

![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.3.0-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-cv2-brightgreen.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)

A Deep learning project for classifying images of cats and dogs using Convolutional Neural Networks (CNN). The project uses data from Kaggle, leverages TensorFlow and Keras for building the CNN model, and utilizes various image processing libraries and techniques.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)



## Introduction
This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN). The primary objective is to develop a model that can accurately distinguish between images of cats and dogs. The dataset for this project is sourced from Kaggle and consists of thousands of labeled images. By leveraging TensorFlow and Keras, this project builds, trains, and evaluates a CNN model for this classification task.

## Features
- **CNN for Image Classification:** Implements a Convolutional Neural Network to classify images of cats and dogs.
- **Data Augmentation:** Uses `ImageDataGenerator` from Keras to augment the training data, improving the model's generalization capabilities.
- **Image Processing:** Utilizes PIL and OpenCV (cv2) for image preprocessing tasks.
- **File Handling:** Employs `os` functions to manage file and directory operations.
- **Custom Batching:** Creates custom data batches for efficient training.
- **Kernel Visualization:** Provides visualizations of the kernels (filters) within the model.
- **Performance Evaluation:** Evaluates model performance using a confusion matrix and accuracy metrics.

## Model Architecture 
Conv2D Layer

The formula to calculate the number of parameters in a Conv2D layer is:
Parameters=(kernel_height×kernel_width×input_channels+1)×number_of_filters
The "+1" accounts for the bias term for each filter.
1. First Conv2D Layer
Input shape: (150, 150, 3)
Number of filters: 32
Filter size: (3, 3)
Parameters:
(3×3×3+1)×32=896
2. Second Conv2D Layer
Input channels: 32 (from the previous Conv2D layer)
Number of filters: 64
Filter size: (3, 3)
Parameters:
(3×3×32+1)×64=18,496
3. Third Conv2D Layer
Input channels: 64
Number of filters: 128
Filter size: (3, 3)
Parameters:
(3×3×64+1)×128=73,856
4. Fourth Conv2D Layer
Input channels: 128
Number of filters: 128
Filter size: (3, 3)
Parameters:
(3×3×128+1)×128=147,584
MaxPooling2D Layer

MaxPooling layers do not have parameters; they only reduce the size of the input they're applied to, based on their pool size and stride.
Flatten Layer

The Flatten layer itself doesn't have parameters. It simply reshapes the input but does not affect the total parameter count.
Dense Layer

The formula for a Dense (fully connected) layer is:
Parameters=(input_size+1)×output_size
1. First Dense Layer
Assuming Flatten output size: X (based on the output of the last pooling layer).
Output size: 512
Parameters:
(𝑋+1)×512
2. Second Dense Layer
Input size: 512
Output size: 1 (for binary classification)
Parameters:
(512+1)×1=513




