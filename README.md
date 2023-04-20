
# Reimplementation of Quaternion convolutional neural networks
_Made by group 43, Rens Oude Elferink & Douwe Mulder_



# Introduction
This post writes about our reproduction of the paper "Quaternion convolutional neural networks", by X. Zhu et al [link](http://openaccess.thecvf.com/content_ECCV_2018/html/Xuanyu_Zhu_Quaternion_Convolutional_Neural_ECCV_2018_paper.html).
The blog was commissioned by the Delft University of Technology for the course DeepLearning CS4240 2022/2023. The code of our reproduction can be found here (LINK). It is partially based on the [original repository](https://github.com/XYZ387/QuaternionCNN_Keras) (https://github.com/XYZ387/QuaternionCNN_Keras) and a pytorch implementation of a quaternion network, from [T Parcollet et al](https://github.com/heblol/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_layers.py) (https://github.com/heblol/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_layers.py). Our reimplementation was built in pytorch. With this reproduction we aim to verify the results provided in the paper. Creating reproductions provides added value to the scientific community by identifying possible errors and limitations, thus it helps in advancing the robustness and reliability of scientific papers. The following section will provide an overview of the contents of the paper after which the report continues with the reproduction of the paper.
## Overview of the paper
Convolutional Neural Networks (CNNs) have shown to be powerful tools in the field of Computer Vision. CNNs utilize kernels to extract features from high-dimensional input matrices. However, when dealing with multi-channel inputs such as color images, the conventional approach of merging channels by summing up the convolution results and outputting a single channel per kernel has certain limitations. 

One example of the limitations of the conventional approach is its disregard for the complex interrelationship between different color channels. By simply summing up the outputs for each kernel, important structural information of the color can be lost, resulting in suboptimal representation of the color image. Additionally, the practice of summing up outputs introduces a large number of degrees of freedom for the learning of convolution kernels, which can increase the risk of overfitting, even with heavy regularization terms. Despite these challenges, limited research has been conducted to overcome these issues and develop more effective solutions for color image processing using CNNs.

To tackle the challenges highlighted earlier, the proposed paper introduces a novel approach known as the Quaternion Convolutional Neural Network (QCNN) model, which represents color images in the quaternion domain. Unlike traditional real-valued convolutions that only enforce scaling transformations on the input, the quaternion convolution incorporates both scaling and rotation of the input in the color space, resulting in a more comprehensive structural representation of color information. By leveraging these quaternion-based modules, the authors of the paper are able to establish fully-quaternion CNNs that offer a more effective representation of color images.

The report first discusses the pre-processing, then it continues with the model creation, model evaluation and ends with a discussion and conclusion. The evaluation includes differences of our implementation and missing details of the original paper.

# Dataset
For this reproduction we used the CIFAR-10 dataset as used in the paper as well. This is a dataset consisting of 60.000 32x32 color images with 10 different labels. The labels used in this dataset are airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. This dataset is commonly used to quickly train computer vision models as it is a small dataset with few labels.
## Data pre-processing
The data has been obtained using the CIFAR public dataset from the pytorch dataset library. It was split into a train and testset using the build-in train/test split, which is a split of 83%, 17% respectively. 

Some data pre-processing steps were taken for both the CNN and the QCNN. For the quaternion CNN, a zero matrix is concatenated to each matrix in the original dataset, such that it becomes a four-dimensional matrix. The pre-processing step is shown in the following codesnippet. 


def convert_to_quaternion(batch):
  new_batch = torch.empty(len(batch), 4, 32, 32)
  labels = torch.LongTensor(len(batch))


  # Transform images to quaternion matrices
  for i in range(len(batch)):
    image, label = batch[i][0], batch[i][1]
    real = torch.zeros(32, 32)
    new_image = torch.cat((image, real.unsqueeze(0)), dim=0)
    new_batch[i, :, :, :] = new_image
    labels[i] = label


  return new_batch, labels

As explained in the paper, a transform of the input data is used to boost the performance of the networks. In the paper they describe that they use both shifting and flipping of the input data, which we have then also done using the transform below. The provided code corresponding to this paper only takes into account a horizontal flip and a shift of 0.1, which we have copied as such for our code.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
    transforms.RandomHorizontalFlip(0.5)
])

# Creating the models
To reproduce the results of the paper we need two separate models. First, we need to create a regular CNN, which will function as the baseline for our experiment. Then, we need to recreate the quaternion CNN described in the paper. We base our implementation on the two given repositories on github provided to us: 
https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_ops.py
https://github.com/XYZ387/QuaternionCNN_Keras/blob/master/cifar10_cnn.py)
The first github is a more general quaternion github with the quaternion layers defined in pytorch. In this github an exact QCNN model is missing. In the next github a QCNN model is built and also an experiment is set up with CIFAR-10. However, this depository was written in keras and therefore not duplicatable for out pytorch code. Apart from these githubs we used several sources for similar CIFAR-10 CNN’s. ( Mattioli, G., 2023 February 10)(M, S. 2022). 
This section provides the details for the implementations of the two models.
## Baseline CNN
The baseline CNN, referred to as the "real-valued CNN" in the original report is implemented using a pytorch implementation and it follows the paper's implementation. In the github repository no code for a real-valued CNN was provided. Therefore, most details had to be subtracted from the paper. However, in this paper some details were not discussed in the paper itself, thus had to be taken from the repository of the QCNN model. This included the dropout, ReLu, padding and channel dimensions. For the first and second convolutional layers, the channels were set to 32. For the third and fourth convolutional layers, the size was set to 64. In the report the output size for the 13th layer (the linear layer just after flattening) was set to 512, but the input size was not defined. We calculated that this had to be size 2304. Also note that, similar to the given repository, only the convolutional layers 1 and 7 include padding.

It has the following layers:
Convolutional block 1
Convolutional layer
ReLU
Convolutional layer
ReLu
MaxPool
Dropout (25%)
Convolutional block 2
Convolutional layer
ReLU
Convolutional layer
ReLu
MaxPool
Dropout (25%)

Flatten
Linear layer
ReLU
Dropout (50%)
Linear layer
Softmax

The summary of the model looks like the following:




As can be seen, only the linear layers 1 and 7 have padding. The following code shows the implementation of this model.



```
class CNN(nn.Module):
   """
   3-layer CNN network with max pooling
  
   Args:
       in_channels: number of features of the input image ("depth of image")
       hidden_channels: number of hidden features ("depth of convolved images")
       out_features: number of features in output layer
   """
  
   def __init__(self, in_channels: int, hidden_channels, out_features: int, kernel_size: int):
       super(CNN, self).__init__()


       self.net = nn.Sequential(
         # block 1
         nn.Conv2d(in_channels, hidden_channels[0], kernel_size, padding=1),
         nn.ReLU(),
         nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size),
         nn.ReLU(),
         nn.MaxPool2d(2, stride=2),
         torch.nn.Dropout(p=0.25),


         # block 2
         nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size, padding=1),
         nn.ReLU(),
         nn.Conv2d(hidden_channels[2], hidden_channels[3], kernel_size),
         nn.ReLU(),
         nn.MaxPool2d(2, stride=2),
         torch.nn.Dropout(p=0.25),


         # block 3
         nn.Flatten(),
         nn.Linear(2304, 512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, out_features),
         nn.Softmax()
       )
  
   def accuracy(self, outputs, labels):
     _, preds = torch.max(outputs, dim=1)


     return torch.tensor(torch.sum(preds == labels).item() / len(preds))


   def training_step(self, batch):
     images, labels = batch
     images, labels = images.to(device), labels.to(device)
     out = self(images)                  # Generate predictions
     loss = F.cross_entropy(out, labels) # Calculate loss
     accu = self.accuracy(out,labels)


     return loss, accu


   def validation_step(self, batch):
       images, labels = batch
       images, labels = images.to(device), labels.to(device)
       out = self(images)                    # Generate predictions
       loss = F.cross_entropy(out, labels)   # Calculate loss
       acc = self.accuracy(out, labels)           # Calculate accuracy


       return loss.detach(), acc
  
   def epoch_end(self, epoch, train_acc, train_loss, test_acc, test_loss):
       print("Epoch :",epoch + 1)
       print(f'Train Accuracy:{train_acc*100:.2f}% Validation Accuracy:{test_acc*100:.2f}%')
       print(f'Train Loss:{train_loss:.4f} Validation Loss:{test_loss:.4f}')


   def forward(self, x):
     return self.net.forward(x)
```
## Quaternion CNN
The quaternion CNN (QCNN) was heavily based on the works mentioned in this [github](https://github.com/heblol/Pytorch-Quaternion-Neural-Networks/blob/master/core_qnn/quaternion_layers.py) and this [github](https://github.com/XYZ387/QuaternionCNN_Keras/blob/master/cifar10_cnn.py). The QCNN layers from the first github were used in the reproduction and were assumed to have a correct implementation. The model of the second github is used to implement the QCNN.

The order of the layers is similar to the baseline CNN, but differs in type of layers. Where the CNN uses regular convolutional layers, the QCNN uses quaternion convolutional layers. Additionally, where the CNN uses a regular linear layer, one of these layers is changed to a quaternion linear layer, as given in the quaternion model in the github.

The following codesnippet describes the implementation of the QCNN.



```
class QNN(nn.Module):
   def __init__(self, in_channels, hidden_channels, out_features, kernel_size = 3):
       super(QNN, self).__init__()
  
       self.net = nn.Sequential(
           # Padding
           QuaternionConv(in_channels, hidden_channels[0],
                                    kernel_size, padding=1, stride=1),
           nn.ReLU(),
           QuaternionConv(hidden_channels[0], hidden_channels[1],
                             kernel_size, stride=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           torch.nn.Dropout(p=0.25),


           QuaternionConv(hidden_channels[1], hidden_channels[2],
                             kernel_size, padding=1, stride=1),
           nn.ReLU(),
           QuaternionConv(hidden_channels[2], hidden_channels[3],
                             kernel_size, stride=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           torch.nn.Dropout(p=0.25),


           nn.Flatten(),
           # QDense
           QuaternionLinear(2304, 512),
           nn.ReLU(),
           nn.Dropout(0.5),
           # Dense
           nn.Linear(512, out_features),
           nn.Softmax()
       )
  
   def accuracy(self, outputs, labels):
     _, preds = torch.max(outputs, dim=1)


     return torch.tensor(torch.sum(preds == labels).item() / len(preds))


   def training_step(self, batch):
     images, labels = batch
     images, labels = images.to(device), labels.to(device)
     out = self(images)                  # Generate predictions
     loss = F.cross_entropy(out, labels) # Calculate loss
     accu = self.accuracy(out, labels)


     return loss, accu


   def validation_step(self, batch):
       images, labels = batch
       images, labels = images.to(device), labels.to(device)
       out = self(images)                    # Generate predictions
       loss = F.cross_entropy(out, labels)   # Calculate loss
       acc = self.accuracy(out, labels)           # Calculate accuracy


       return loss.detach(), acc
  
   def epoch_end(self, epoch, train_acc, train_loss, test_acc, test_loss):
       print("Epoch :",epoch + 1)
       print(f'Train Accuracy:{train_acc*100:.2f}% Validation Accuracy:{test_acc*100:.2f}%')
       print(f'Train Loss:{train_loss:.4f} Validation Loss:{test_loss:.4f}')


   def forward(self, x):
     return self.net.forward(x)

```
Weight initialization:
The weights were initialization according to the implementation of X. This was using the `glorot` criterion. The initialization uses a randomly unified distsribution from -1 to + 1.



```
def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):


   if kernel_size is not None:
       receptive_field = np.prod(kernel_size)
       fan_in          = in_features  * receptive_field
       fan_out         = out_features * receptive_field
   else:
       fan_in          = in_features
       fan_out         = out_features


   if criterion == 'glorot':
       s = 1. / np.sqrt(2*(fan_in + fan_out))
   elif criterion == 'he':
       s = 1. / np.sqrt(2*fan_in)
   else:
       raise ValueError('Invalid criterion: ' + criterion)


   rng = RandomState(np.random.randint(1,1234))


   # Generating randoms and purely imaginary quaternions :
   if kernel_size is None:
       kernel_shape = (in_features, out_features)
   else:
       if type(kernel_size) is int:
           kernel_shape = (out_features, in_features) + tuple((kernel_size,))
       else:
           kernel_shape = (out_features, in_features) + (*kernel_size,)


   modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)
   number_of_weights = np.prod(kernel_shape)
   v_i = np.random.uniform(-1.0,1.0,number_of_weights)
   v_j = np.random.uniform(-1.0,1.0,number_of_weights)
   v_k = np.random.uniform(-1.0,1.0,number_of_weights)


   # Purely imaginary quaternions unitary
   for i in range(0, number_of_weights):
       norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
       v_i[i]/= norm
       v_j[i]/= norm
       v_k[i]/= norm
   v_i = v_i.reshape(kernel_shape)
   v_j = v_j.reshape(kernel_shape)
   v_k = v_k.reshape(kernel_shape)


   phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)


   weight_r = modulus * np.cos(phase)
   weight_i = modulus * v_i*np.sin(phase)
   weight_j = modulus * v_j*np.sin(phase)
   weight_k = modulus * v_k*np.sin(phase)


   return (weight_r, weight_i, weight_j, weight_k)
```
## Training Procedure
Training was done in batches of 32, with 80 epoch, learning rate of 0.0001. This was according to the paper. The repository has a learning decay of 0.000001, however, it was not mentioned in the paper how exactly this was implemented. During experiments the results with learning decay were significantly worse and therefore we decided not to use them in the final model. In the paper and the keras code RMSprop was used for the optimizer and cross entropy for the loss function. Therefore we decided also to use these for both our models.



# Results / Evaluation
This section describes the results obtained by us using the described models. In addition, this section compares our results with the results from the original paper. After running the models, we got the results given in the figure below.







Training loss in paper (above) and in our experiments (below).

Test accuracy in paper (above) and in our experiments (below).




In this figure we can see that the training loss learning curve in the paper is much steeper than the curve obtained in our experiments. Moreover, the final obtained loss is much lower in the paper than in our experiments. What is curious to note about this is that even though the loss is much higher, the accuracy remains similar to the paper. 

The two figures on the right show the accuracies obtained by us (figure below) and by the paper (figure above). Here it can be seen that the baseline models perform similarly, while the QCNN in our experiment performs much worse than in the paper. The testing accuracies of the paper and our experiment are given below.





These tables show that our baseline works better than the baseline in the paper, with a difference of almost 2 percentage-points. The QCNN implemented by us performs worse than the QCNN in the paper, with a difference of over 7 percentage-points.

The next section will discuss the results obtained by our experiments and it will give explanations for the differences found in the models.
# Discussion / Conclusion:
The goal of the blog was to reproduce an experiment from the paper ‘Quaternion Convolutional Neural Networks’. The original paper was implemented using an old version of tensorflow and therefore the pytorch implementation of T. Parcollet et. al. was used to implement the quaternion layers instead. We made the assumption that this implementation was similar to the implementation used in the paper. With these quaternion layers we could implement the network given in the original code. The reimplementation focused on pre-processing the dataset for the quaternion model, building the baseline "real-valued" CNN and building the QCNN following the layout of the paper. After the creation of these models we ran the experiments on the CIFAR dataset.

The results show that our baseline model performs similar to the baseline in the paper. This assures us that the baseline implementation and training loop work as expected. However, the quaternion neural network performs significantly worse than mentioned in the paper. With a percentage-point difference of 7 points we can conclude that we were not able to reproduce the results given in the paper. There are multiple aspects that could have caused this.

The difference is probably caused by a difference in the implementation of the quaternion layers between the original paper and the repository consulted by us. It is difficult to exactly pinpoint the main difference between these implementations, since both implementations have a completely different approach to defining the layers. Additional research would be needed to find the exact differences between them. A first intuition would suggest that the difference can be caused by the way the input is handled. In the paper the input is expected to be a three-channel matrix, corresponding to the rgb-channels of an image. In contrast, the consulted repository used a four-dimensional input matrix, where they did not extensively specify how they created this four-dimensional data from a rgb-image. It could be that we made a mistake in the conversion from rgb to quaternion. 

Another difference between the repository and the paper is the way that the weights of the quaternion network are initialized. The paper provides details of the weight initialization, but these do not seem to match the weight initialization given in the repository. For us it was not clear how to implement the technique in the paper, therefore we did not use it. The weight initialization can be of great impact to the training of the network and, more specifically, to the training speed. As we can see in the learning curve, the QCNN was not done training at epoch 80. This could be caused by the weights not being intialized properly at the start, thus slowing down the training. To counter this effect you could extend the learning time with more epochs. However, due to limited resources we were not able to execute this. It remains unclear how good the QCNN could perform if the training would have been finished.

If we look back at the process we went through to reproduce the results, we can make some comments on the difficulty to reproduce the paper. Firstly, we appreciated that the authors made their code available for others. However, this code was out-of-date due to an old version of Tensorflow. This, of course, makes it harder for others to reproduce. We would therefore like to suggest others to keep their code up-to-date, such that others will always be able to easily reproduce it. We would also like to add that it would have been useful if the authors had provided the code used for the experiments. This way there would have been no confusion whether the implementation of the experiments was correct. One small thing we would also like to add is that some hyperparameters were not clearly defined in the paper. For example, the learning rate decay was given as a constant, but it was not defined how this constant should be used to decay the learning rate.

To conclude, we were not able to reproduce the results mentioned in the paper ‘Quaternion Convolutional Neural Networks’. Our baseline implementation was on-par with the baseline mentioned in the paper. However, the QCNN performed significantly worse than presented in the paper. This is probably caused by a difference in implementation of the quaternion layers between the two github repositories used by us.













# References
Mattioli, G. (2023, February 10). CIFAR10 Image Classification in PyTorch | by Gabriele Mattioli | MLearning.ai. Medium. https://medium.com/mlearning-ai/cifar10-image-classification-in-pytorch-e5185176fbef

M, S. (2022). Convolutional Neural Network – PyTorch implementation on CIFAR-10 Dataset. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2021/09/convolutional-neural-network-pytorch-implementation-on-cifar10-dataset/



