# Image classification
image classifier including lenet, vgg, resnet, inception

## Pre-requisite
tensorflow 1.12
numpy

## Setup
	cd $PROJECT_DIR$
	export PYTHONPATH=PYTHONPATH:`pwd`

## Usage
if you want to train mnist,you can run like this:

	python example/mnist/train_eval_mnist.py

if you want to train cifar,you can run like this:

	python dataset/cifar10_download_and_extract.py
	python example/cifar10/train_eval_cifar.py

I got ninety-three precent accurancy based on cifar-10

## Reference
The project is based on [tensorflow/models](https://github.com/tensorflow/models)
