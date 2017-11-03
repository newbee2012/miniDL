#include "common.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "net_model.hpp"
#include "input_layer.hpp"
#include "conv_layer.hpp"
#include "layer.hpp"
using namespace std;

namespace dong
{

void NetModel::forward()
{
    cout<<"NetModel forward..."<<endl;
    boost::shared_ptr<Layer> layer = _inputLayer;
    do
    {
        cout<<"forward "<<layer->getName()<<endl;
        layer->forward();
        layer = layer->getTopLayer();

    }
    while(layer.get());
}

void NetModel::setUpInputLayer()
{
    int shape_size = _input_shape_num*_input_shape_channels*_input_shape_height * _input_shape_width;
    ASSERT(shape_size>0, cout<<"训练数据尺寸定义错误！"<<endl);

    _inputNeurons.reset(new Neuron[_input_shape_num *_input_shape_channels *_input_shape_height * _input_shape_width]);
    _inputData.reset(new Data(_input_shape_num, _input_shape_channels, _input_shape_height, _input_shape_width));
    _inputData->setUp(_inputNeurons);
    _inputLayer->setUp(_inputData);
}

void NetModel::fillDataForOnceTrainForward(Neuron* datas, int size)
{
    int shape_size = _input_shape_num*_input_shape_channels*_input_shape_height * _input_shape_width;
    ASSERT(size >= shape_size, "输入数据size<shape_size");
    for (int i = 0; i < shape_size; ++i)
    {
        _inputNeurons[i]._value = (*(datas++))._value;
        cout<<(*(datas++))._value<<",";
    }
    cout<<endl;
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

    _per_iter_train_count = jo_hyperParameters["perIterTrainCount"].asInt();
    _batch_count = jo_hyperParameters["batchCount"].asInt();

    Json::Value jo_input_shape = root["inputShape"];
    ASSERT(!jo_input_shape.isNull(), cout<<"节点inputShape不存在！"<<endl);

    _input_shape_num = jo_input_shape["num"].asInt();
    _input_shape_channels = jo_input_shape["channels"].asInt();
    _input_shape_height = jo_input_shape["height"].asInt();
    _input_shape_width = jo_input_shape["width"].asInt();
    cout<<_input_shape_num<<endl;

    Json::Value jo_layers = root["layersModel"];
    cout<<"model json:"<<endl<<jo_layers<<endl;
    ASSERT(!jo_layers.isNull(), cout<<"节点layersModel不存在！"<<endl);

    //查找输入层
    Json::Value jo_layer = jo_layers["inputLayer"];
    ASSERT(!jo_layer.isNull(), cout<<"inputLayer不存在！"<<endl);

    Layer* bottom_layer = NULL;

    while(!jo_layer.isNull())
    {
        string type = jo_layer["type"].asString();
        const string name = jo_layer["name"].asString();
        const string top_layer_name = jo_layer["topLayer"].asString();
        Json::Value init_params = jo_layer["initParams"];
        int params_size = init_params.size();
        int params[4] = {0};
        cout<<"type:"<<type<<endl;
        cout<<"name:"<<name<<endl;
        cout<<"top_layer_name:"<<top_layer_name<<endl;
        cout<<"params:";
        for(int j = 0; j < params_size; ++j)
        {
            params[j] = init_params[j].asInt();
            cout<<params[j]<<",";
        }

        cout<<endl;
        Layer* layer = NULL;
        switch (STRING_TO_LAYER_TYPE(type.c_str()))
        {
        case INPUT_LAYER:
            layer = new InputLayer(name);
            _inputLayer.reset(layer);
            this->setUpInputLayer();
            break;
        case CONVOLUTION_LAYER:
            layer = new ConvLayer(name);
            break;
        default:
            ASSERT(false, cout<<"不合法的LayerType-->"<<type.c_str()<<endl);
            break;
        }

        layer->init(params);
        if(NULL != bottom_layer)
        {
            bottom_layer->setTopLayer(layer);
        }

        bottom_layer = layer;
        //_layers.push_back(boost::shared_ptr<Layer> (layer));
        jo_layer = jo_layers[top_layer_name];

        cout<<endl;
    }

    is.close();
}

}
