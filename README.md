# MiniDL
# A minimalist C++ Deep-Learning framework
# 从零开始用c++编写一个教学目的深度学习框架
#### 1.可用JSON快速定义简单的神经网络模型，训练模型，验证模型。例如: MNIST，cifar-10
#### 2.代码实现没有引用任何数学库，甚至没有通常的矩阵运算，初学者甚至不用了解反向传播算法的数学公式推导.
#### 3.最初目的是帮助学习者理解反向传播算法本质:上下层神经元之间的误差传递和修正。
1.How to build the project?
sh ./build.sh

2.How to define a model?
For exsample : net_model_define_mnist.json

3.How to run & train a model?
(1) Modify the value of the "model" field in the model definition JSON file to "train"
(2) ./bin/Release/miniDL net_model_define_mnist.json

4.How to run & test a model?
(1) Modify the value of the "model" field in the model definition JSON file to "test"
(2) ./bin/Release/miniDL net_model_define_mnist.json
