# Reimplementation of Quaternion convolutional neural networks
_Made by group 43, Rens oude Elferink & Douwe Mulder_

# Introduction
This post writes about our reproduction of the paper "Quaternion convolutional neural networks", by X. Zhu et al [link](http://openaccess.thecvf.com/content_ECCV_2018/html/Xuanyu_Zhu_Quaternion_Convolutional_Neural_ECCV_2018_paper.html).
The blog was commissioned by the Delft University of Technology for the course DeepLearning CS4240 2022/2023. The code of our reproduction can be found here (LINK). It is partially based on the original repository [link](https://github.com/XYZ387/QuaternionCNN_Keras) (https://github.com/XYZ387/QuaternionCNN_Keras) and a pytorch implementation of a quaternion network, from T Parcollet et al [link](https://github.com/heblol/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_layers.py) (https://github.com/heblol/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_layers.py). Our reimplementation was built in pytorch. The following section will provide an overview of the contents of the paper.

## Overview of the paper
Convolutional Neural Networks (CNNs) have shown to be powerful in the field of Computer Vision.  One key module of CNN model is the convolution layer, which extracts features from high-dimensional structural data efficiently by a set of convolution kernels. When dealing with multi-channel inputs (e.g., color images), the convolution kernels merges these channels by summing up the convolution results and output one single channel per kernel accordingly. While this approach performs well in many scenarios, it inherently suffers from certain limitations.

Firstly, this approach ignores the complex interrelationship between different color channels, as it simply sums up the outputs for each kernel. This can result in the loss of important structural information of the color, leading to suboptimal representation of the color image.

Secondly, the practice of summing up outputs introduces a large number of degrees of freedom for the learning of convolution kernels, which increases the risk of overfitting, even with heavy regularization terms.

Despite these challenges, there has been limited investigation into how to overcome these issues and develop more effective solutions for color image processing using CNNs.

In order to address the challenges mentioned earlier, the considered paper proposes a novel approach called the Quaternion Convolutional Neural Network (QCNN) model, which represents color images in the quaternion domain. Unlike traditional real-valued convolutions that only enforce scaling transformations on the input, the quaternion convolution achieves both scaling and rotation of the input in the color space, providing a more comprehensive structural representation of color information. By leveraging these quaternion-based modules, the authors of the paper can establish fully-quaternion CNNs that offer more effective representation of color images.

The report follows the implementation structure, i.e., first discusses the pre-processing, model creation, model evaluation and ends with a discussion and conclusion. The evaluation includes differences of our implementation and missing details of the original paper.
