# mnist-digit-classification

Classifying images in the [MNIST](http://yann.lecun.com/exdb/mnist/) database for handwritten digits using the [TensorFlow](https://www.tensorflow.org/) library with Python.

## Models
### Flat model

The flat model consists of a simple feedforward neural network with 8 layers. The input layer contains 784 nodes which represent the flattened image of a handwritten digit. The output layer consists of 10 nodes and the `argmax` of the output layer is deemed as the predicted digit.


This model was trained for **100 epochs** and achieved an accuracy of **97.46%** on the test image dataset.

#### Examples

|Correct classifications|![5](http://i.imgur.com/QOwbQDV.png)|![2](http://i.imgur.com/yImnAXp.png)|![0](http://i.imgur.com/pIfZmnW.png)|![6](http://i.imgur.com/PMGHJVN.png)|![4](http://i.imgur.com/qMxs7ha.png)|
|---|:-:|:-:|:-:|:-:|:-:|
|Predicted|5|2|0|6|4|
|Actual|5|2|0|6|4|

|Incorrect classifications|![1](http://i.imgur.com/tTsIRWy.png)|![5](http://i.imgur.com/u8W1JSi.png)|![0](http://i.imgur.com/JRiok6v.png)|![8](http://i.imgur.com/1lXNlfB.png)|![8](http://i.imgur.com/JFykBhE.png)|
|---|:-:|:-:|:-:|:-:|:-:|
|Predicted|7|8|2|2|7|
|Actual|1|5|0|8|8|

Although the model has a relatively high accuracy, it fails to generalize well and misclassified some images that are easily legible due to the structure and approach to the problem (flattening the image).
