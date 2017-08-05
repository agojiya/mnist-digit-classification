# mnist-digit-classification

Classifying images in the [MNIST](http://yann.lecun.com/exdb/mnist/) database for handwritten digits using the [TensorFlow](https://www.tensorflow.org/) library with Python.

## Models
### conv2d model

The conv2d model consists of four pairs of `conv2d`-`max_pooling2d` layers connected to two `dense` layers, each with 512 nodes. The output of the last `dense` layer is normalized such that the output of each node lies in between `-1` and `+1`. The normalized output is fed into the output layer (`dense`, 10 nodes) which applies a `sigmoid` activation function. The `argmax` of the activated outputs of the output layer is deemed as the predicted digit.

This model was trained for **100 epochs** and achieved an accuracy of **99.32%** on the test image dataset.

#### Examples

|Correct classifications|![4](http://i.imgur.com/O7Ktm4b.png)|![0](http://i.imgur.com/Z6wLljH.png)|![1](http://i.imgur.com/3UnKo6Q.png)|![8](http://i.imgur.com/2FV0Urm.png)|![7](http://i.imgur.com/aQv91Uu.png)|
|---|:-:|:-:|:-:|:-:|:-:|
|Predicted|4|0|1|8|7|
|Confidence|100.00%|100.00%|100.00%|100.00%|100.00%|
|Actual|4|0|1|8|7|

|Incorrect classifications|![6](http://i.imgur.com/QlgWWHC.png)|![0](http://i.imgur.com/hNKjVhS.png)|![8](http://i.imgur.com/dZtefWv.png)|![2](http://i.imgur.com/eNSdvBr.png)|![5](http://i.imgur.com/RH6lRO6.png)|
|---|:-:|:-:|:-:|:-:|:-:|
|Predicted|0|8|2|0|3|
|Confidence|79.08%|30.70%|98.27%|87.85%|91.36%|
|2<sup>nd</sup> best prediction|6|6|8|2|5|
|Actual|6|0|8|2|5|

_The second best prediction is simply the `argmax` of the output with the initial prediction ignored._

The model has a very high accuracy because it can successfully recognize features such as edges and shapes, and use them to come to a conclusion. The misclassifications are mainly skewed, rotated, or otherwise abnormal (ie: Misclassification #4).

To improve accuracy, the images can be cleaned up (rotation, skew correction, growth/decay of pixels towards the average\*, etc.) before they are provided to the network during both training and testing.

\* This would help by normalizing images where the numbers are too narrow or bold

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
