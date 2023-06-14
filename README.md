# CNN-for-Image-Classification
A model is implemented for the classification of CIFAR-10 dataset images using a deep-learning programming framework. This data set consists of 60,000 color images with dimensions of 32x32 (including 50,000 training images and 10,000 test images). The images of this data set are divided into ten categories.

Consider a convolutional network with the following architecture:
1) Convolutional algorithm: kernel size: 3x3, (number of output channels: 7 and activation function: ReLU)
2) Convolutional algorithm: kernel size: 3x3, (number of output channels: 9 and activation function: ReLU)
3) God of maximum integration: Kernel size: 2x2
4) Out-drop layer with a 30% probability
5) Linear layer with the number of ten (number of groups) output neurons

This network was trained with the Adam optimizer, cross-categorical entropy error function, and a learning rate of 100% (with a batch size of 32).
