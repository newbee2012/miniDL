# MiniDL
# A minimalist C++ Deep-Learning framework
# 从零开始用c++编写一个教学目的深度学习框架
#### 1.可用JSON快速定义简单的神经网络模型，训练模型，验证模型。例如: MNIST，cifar-10
#### 2.代码实现没有引用任何数学库，甚至没有通常的矩阵运算，初学者甚至不用了解反向传播算法的数学公式推导.
#### 3.最初目的是帮助学习者理解反向传播算法本质:上下层神经元之间的误差传递和修正。

1.How to define a model?

For exsample : mnist JSON model definition
{
    "mode":"TRAIN",
    "modelDataFilePath": "./mnist_model_data.json",
    "trainDataFilePath": "./mnist_train_lmdb",
    "testDataFilePath": "./mnist_test_lmdb",
    "initModelByExistentData": false,
    "hyperParameters": {
        "MAX_ITER_COUNT": 1,
        "BATCH_SIZE": 1,
        "BASE_LEARNING_RATE": 0.01,
        "LEARNING_RATE_POLICY": "INV",
        "GAMMA": 0.0001,
        "MOMENTUM": 0.9,
        "POWER": 0.75,
        "WEIGHT_DECAY": 0.0005,
        "STEPSIZE": 100,
	"FORWARD_THREAD_COUNT": 1,
	"BACKWARD_THREAD_COUNT": 1
    },
    "inputShape": {
        "channels": 1,
        "height": 28,
        "width": 28
    },
    "layersModel": {
	"inputLayer": {
            "implClass": "InputLayer",
            "initParams": null,
            "topLayer": "convLayer",
	    "scale": 1
        },
        "convLayer": {
            "implClass": "ConvLayer",
            "initParams": [20,5,5,0,0,1],
            "topLayer": "maxPoolLayer",
	    "lr_mult_weight": 1,
	    "lr_mult_bias": 2,
	    "weight_init": {
		"type":"CONSTANT",
		"constant_value":1
	    },
	    "bias_init": {
		"type":"CONSTANT",
		"constant_value":0
	    }
        },
	"maxPoolLayer": {
            "implClass": "AvePoolLayer",
            "initParams": [2,2,2,2],
            "topLayer": "convLayer2",
	    "lr_mult_weight": 1,
	    "lr_mult_bias": 2
        },
	"convLayer2": {
            "implClass": "ConvLayer",
            "initParams": [50,5,5,0,0,1],
            "topLayer": "maxPoolLayer2",
	    "lr_mult_weight": 1,
	    "lr_mult_bias": 2,
	    "weight_init": {
		"type":"XAVIER"
	    },
	    "bias_init": {
		"type":"CONSTANT",
		"constant_value":0
	    }
        },
	"maxPoolLayer2": {
            "implClass": "MaxPoolLayer",
            "initParams": [2,2,2,2],
            "topLayer": "fullConnectLayer",
	    "lr_mult_weight": 1,
	    "lr_mult_bias": 2
        },
        "fullConnectLayer": {
            "implClass": "FullConnectLayer",
            "initParams": [500],
            "topLayer": "reluLayer",
	    "lr_mult_weight": 1,
	    "lr_mult_bias": 2,
	    "weight_init": {
		"type":"XAVIER"
	    },
	    "bias_init": {
		"type":"CONSTANT",
		"constant_value":0
	    }
        },
        "reluLayer": {
            "implClass": "ReluLayer",
            "initParams": [],
            "topLayer": "fullConnectLayer2"
        },
	"fullConnectLayer2": {
            "implClass": "FullConnectLayer",
            "initParams": [10],
            "topLayer": "lossLayer",
	    "lr_mult_weight": 1,
	    "lr_mult_bias": 2,
	    "weight_init": {
		"type":"XAVIER"
	    },
	    "bias_init": {
		"type":"CONSTANT",
		"constant_value":0
	    }
        },
        "lossLayer": {
            "implClass": "SoftmaxLayer",
            "initParams": [],
            "topLayer": null
        }
    }
}
