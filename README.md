# MiniDL
# A minimalist C++ Deep-Learning framework

1. A simple neural network model can be quickly defined by JSON, and the model can be trained and verified. For example: MNIST, cifar-10
2. The implementation of the code does not refer to any mathematical library, or even the usual matrix operation. Beginners do not even need to understand the mathematical formula derivation of the BP algorithm
3. The original purpose is to help learners understand the essence of BP algorithm: error transmission and correction between upper and lower neurons.

___
# 从零开始用C++编写一个教学目的深度学习框架
#### 1.可用JSON快速定义简单的神经网络模型，训练模型，验证模型。例如: MNIST，cifar-10
#### 2.代码实现没有引用任何数学库，甚至没有通常的矩阵运算，初学者甚至不用了解反向传播算法的数学公式推导.
#### 3.最初目的是帮助学习者理解反向传播算法本质:上下层神经元之间的误差传递和修正。

___
+ How to build the project?
  + *sh ./build.sh*

+ How to define a model?
  + *For exsample : ./net_model_define_mnist.json*

+ How to run & train a model?
  - *Modify the value of the "model" field in the model definition JSON file to "train"*
  - *./bin/Release/miniDL net_model_define_mnist.json*

+ How to run & test a model?
  - *Modify the value of the "model" field in the model definition JSON file to "test"*
  - *./bin/Release/miniDL net_model_define_mnist.json*
