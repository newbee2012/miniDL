#include "common.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "net_model.hpp"
#include "softmax_layer.hpp"
#include "pool_layer.hpp"
#include "input_layer.hpp"
#include "conv_layer.hpp"
#include "full_connect_layer.hpp"
#include "relu_layer.hpp"

#include "layer.hpp"
using namespace std;

namespace dong
{

void NetModel::forward()
{
    //cout<<"NetModel forward..."<<endl;
    boost::shared_ptr<Layer> layer = _input_layer;
    while(layer->getTopLayer().get())
    {
        layer = layer->getTopLayer();
        //cout<<"forward "<<LAYER_TYPE_TO_STRING(layer->getType())<<endl;
        layer->forward();
        if(layer->getType() == LOSS_LAYER)
        {
            LossLayer* lossLayer = (LossLayer*)layer.get();
            cout<<"Loss:"<<lossLayer->getLoss()<<endl;
        }
    }
}

void NetModel::setUpInputLayer()
{
    int shape_size = _input_shape_num*_input_shape_channels*_input_shape_height * _input_shape_width;
    ASSERT(shape_size>0, cout<<"训练数据尺寸定义错误！"<<endl);

    _input_neurons.reset(new Neuron[_input_shape_num *_input_shape_channels *_input_shape_height * _input_shape_width]);
    _input_data.reset(new Data(_input_shape_num, _input_shape_channels, _input_shape_height, _input_shape_width));
    _input_data->setUp(_input_neurons);
    _input_layer->setUp(_input_data);
}

void NetModel::fillDataForOnceTrainForward(Neuron* datas, int size, int label)
{
    int shape_size = _input_shape_num*_input_shape_channels*_input_shape_height * _input_shape_width;
    ASSERT(size >= shape_size, "输入数据size<shape_size");
    for (int i = 0; i < shape_size; ++i)
    {
        _input_neurons[i]._value = datas[i]._value;
    }

    if(_loss_layer.get() != NULL)
    {
       //cout<<"setLabel:"<<label<<endl;
       _loss_layer->setLabel(label);
    }

}

Layer* NetModel::generateLayerByClassName(const char* className)
{
    if(0==strcmp(className,"InputLayer"))
    {
        return new InputLayer();
    }
    else if(0==strcmp(className,"ConvLayer"))
    {
        return new ConvLayer();
    }
    else if(0==strcmp(className,"FullConnectLayer"))
    {
        return new FullConnectLayer();
    }
    else if(0==strcmp(className,"ReluLayer"))
    {
        return new ReluLayer();
    }
    else if(0==strcmp(className,"PoolLayer"))
    {
        return new PoolLayer();
    }
    else if(0==strcmp(className,"SoftmaxLayer"))
    {
        return new SoftmaxLayer();
    }

    return NULL;
}

void NetModel::save_model(const char* filename)
{

}

void NetModel::load_model(const char* filename)
{
    cout<<"loading model...."<<endl;
    // 解析json用Json::Reader
    Json::Reader reader;
    // Json::Value是一种很重要的类型，可以代表任意类型。如int, string, object, array...
    Json::Value root;

    std::ifstream is;
    is.open (filename, std::ios::binary );
    if (!reader.parse(is, root))
    {
        ASSERT(false, cout<<"Json 解析失败！"<<endl);
    }

    Json::Value jo_hyperParameters = root["hyperParameters"];
    ASSERT(!jo_hyperParameters.isNull(), cout<<"节点hyperParameters不存在！"<<endl);

    _per_batch_train_count = jo_hyperParameters["perBatchTrainCount"].asInt();
    ASSERT(_per_batch_train_count>0, cout<<"perBatchTrainCount必须大于0！"<<endl);
    _batch_count = jo_hyperParameters["batchCount"].asInt();
    ASSERT(_batch_count>0, cout<<"batchCount必须大于0！"<<endl);

    Json::Value jo_input_shape = root["inputShape"];
    ASSERT(!jo_input_shape.isNull(), cout<<"节点inputShape不存在！"<<endl);

    _input_shape_num = jo_input_shape["num"].asInt();
    _input_shape_channels = jo_input_shape["channels"].asInt();
    _input_shape_height = jo_input_shape["height"].asInt();
    _input_shape_width = jo_input_shape["width"].asInt();
    Json::Value jo_layers = root["layersModel"];
    cout<<"model json:"<<endl<<jo_layers<<endl;
    ASSERT(!jo_layers.isNull(), cout<<"节点layersModel不存在！"<<endl);

    //查找输入层
    Json::Value jo_layer = jo_layers["inputLayer"];
    ASSERT(!jo_layer.isNull(), cout<<"inputLayer不存在！"<<endl);

    boost::shared_ptr<Layer> bottom_layer;

    while(!jo_layer.isNull())
    {
        const string top_layer = jo_layer["topLayer"].asString();
        const string impl_class = jo_layer["implClass"].asString();
        Json::Value init_params = jo_layer["initParams"];
        int params_size = init_params.size();
        int params[4] = {0};
        cout<<"impl_class:"<<impl_class<<endl;
        cout<<"top_layer:"<<top_layer<<endl;
        cout<<"params:";
        for(int j = 0; j < params_size; ++j)
        {
            params[j] = init_params[j].asInt();
            cout<<params[j]<<",";
        }

        cout<<endl;

        boost::shared_ptr<Layer> layer(generateLayerByClassName(impl_class.c_str()));
        layer->init(params);
        if(layer->getType() == INPUT_LAYER)
        {
            _input_layer = layer;
            this->setUpInputLayer();
        }else if(layer->getType() == LOSS_LAYER)
        {
            _loss_layer = layer;
        }


        if(NULL != bottom_layer)
        {
            bottom_layer->setTopLayer(layer);
        }

        bottom_layer = layer;
        jo_layer = jo_layers[top_layer];

        cout<<endl;
    }

    is.close();
}

}
