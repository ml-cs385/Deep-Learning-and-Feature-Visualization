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

### More Detailed Tasks

1. 写每个数据集对应的DataLoader文件

   - MNIST
   - CIFAR-10
   - CUB200-2011
   - Pascal VOC 2012
   - Stanford Dogs

   每个DataLoader应该有的接口是：

   - load_dataset：制作数据集
     - 得到train_images, train_labels, test_images, test_labels，可以取出整个训练集和测试集
     - 相应写出save_to_pickle, load_pickle_dataset：制作数据集后直接用pickle存储后直接载入就好了，不用重复制作
   - next_batch_train, next_batch_test: 输出一个batch的训练数据和测试数据

   - resize_images: 重新设置image的大小，方便计算

   - 其余的接口根据任务来添加，可在做得过程中添加

2. Image Classification

   - 寻找下载网上已经训好的预训练模型，包括：AlexNet, VGG-16/19, ResNets
     - 注：VGG-16有一个准确率不是很高的模型，ResNet在Repo里面有一个模型，最好找准确率高的，用大数据集（ImageNet）训练过的模型
   - 保留原模型的卷积层，根据不同数据集调整模型的全连接层，进行训练，记录准确率
     - AlexNet, VGG-16/19, ResNets
   - 保留原模型前面若干层，修改后面的层（试着修改卷积层），观察会有什么效果，记录准确率
     - Modified AlexNet, VGG-16/19, ResNets

   每个网络应该有的接口：

   - build_model: 搭建模型
   - load_model: 载入预训练模型的参数，这里根据自己的设定，只载入保留下来层的参数，其余层的参数正常初始化
   - train_model: 输入数据训练模型，返回（或者在函数中打印也行，建议返回，因为不是每一次都要打印）loss和train_accracy
     - 要求中写到尝试不同地learning_rate，要写一个参数指定是什么样的learning rate（是否是descending的）
   - test_acc: 测试数据的准确率，打印test_accuracy
   - get_image_features：返回特定层的图片输出feature
     - 可以有一个参数layer_num: 指定哪层
   - 其余接口根据任务添加

   所以这个任务的目标是：不同网路（包括原网络和修改后的网络）、不同数据集下的准确率、不同learning_rate设置下的训练过程表现（可以画图画出训练过程中loss和accuracy是怎么变化的）

3. Feature Analysis

   - PCA: 

     - 抽取训练后模型的feature，使用get_image_features
     - 降维
     - 聚类并可视化

     此任务尝试不同网络、不同层的feature的，不同降维维度的聚类效果（三个影响因子）

   - t-SNE:

     - 降维可视化PCA降维后的feature
     - 降维可视化原始feature

     此任务尝试不同网络、不同层的feature的，不同降维维度的聚类效果（三个影响因子）+ 是否PCA

   - Grad-Cam:

     - 不了解，需要学习一下
     - 可视化不同层的activation

     实验就做不同网络、不同层的activation

4. Image Segmentation

   - 不了解，待我慢慢看一下

5. Image Reconstruction（Mengqi自己写一下）

   - To be written