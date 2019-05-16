# Note
network architecture, data preprocessing, related paper for project 2

根据的代码改写了TensorFlow-gpu 1.4版本的ResNet，目前可以运行

# CS385 Pipeline

## Dataset (Recommended)

##### Classificaiton

- MNIST
- CIFAR-10

##### Object Detection

- CUB200-2011

- Pascal VOC 2012
- Stanford dogs datasets

## Task

#### Multi-category classification 多分类问题

要求:

- AlexNet, VGG-16/19, ResNets

- PCA + clustering + t-SNE 对特征进行分析
  - Factors that affect the clustering property of intermediate-layer features
- 调整网络结构和超参 (这些可以画一个图横轴是对应的超参纵轴是accuracy)
  - Layer number
  - Kernel number per layer
  - Kernel size
  - Set a constant/descending learning rate
- Grad-CAM 是篇论文, G站上Keras和PyTorch的版本都有
  - Factors that affect the attention (grad-CAM) of a neural network

> https://arxiv.org/abs/1610.02391 , ICCV 2017

#### Image segmentation: To classify the category label for each pixel (包括一个background类)

要求: Modify the structure of the AlexNet/VGG/ResNet

####  Image reconstruction 感觉就是个Auto-Encoder

要求:  Variational Auto-Encoder