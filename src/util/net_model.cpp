#include "common.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "util/net_model.hpp"
#include "input_layer.hpp"
#include "conv_layer.hpp"
using namespace std;

namespace dong
{

void NetModel::setUp()
{
    int shape_size = _input_shape_num*_input_shape_channels*_input_shape_height * _input_shape_width;
    ASSERT(shape_size>0, cout<<"训练数据尺寸定义错误！"<<endl);

    _inputNeurons.reset(new Neuron[_input_shape_num*_input_shape_channels*_input_shape_height * _input_shape_width]);
    _inputData.reset(new Data(_input_shape_num, _input_shape_channels, _input_shape_height, _input_shape_width));
    _inputData->setUp(_inputNeurons);
    _inputLayer->setUp(_inputData);
}

void NetModel::train()
{

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
    if (reader.parse(is, root))
    {
        Json::Value input_shape = root["inputShape"];
        ASSERT(!input_shape.isNull(), cout<<"节点inputShape不存在！"<<endl);

        _input_shape_num = input_shape["num"].asInt();
        _input_shape_channels = input_shape["channels"].asInt();
        _input_shape_height = input_shape["height"].asInt();
        _input_shape_width = input_shape["width"].asInt();
        this->setUp();

        Json::Value layers = root["layersModel"];
        cout<<"model json:"<<endl<<layers<<endl;
        ASSERT(!layers.isNull(), cout<<"节点layersModel不存在！"<<endl);

        int layers_size = layers.size();
        ASSERT(layers_size > 0, cout<<"layers_size <=0！"<<endl);
        _layers.reset(new Layer*[layers_size]);

        for(int i = 0; i < layers_size; ++i)
        {
            cout<<"layer["<<i<<"]:"<<endl;
            Json::Value layer = layers[i];
            string type = layer["type"].asString();
            string name = layer["name"].asString();
            string bottom_layer_name = layer["bottomLayerName"].asString();

            switch (STRING_TO_LAYER_TYPE(type.c_str()))
            {
            case CONVOLUTION_LAYER:
                _layers[i] = new ConvLayer();
                break;
            default:
                ASSERT(false, cout<<"不合法的LayerType-->"<<type.c_str()<<endl);
                break;
            }

            Json::Value init_params = layer["initParams"];
            int params_size = init_params.size();
            int params[4] = {0};
            cout<<"params:";
            for(int j = 0; j < params_size; ++j)
            {
                params[i] = init_params[j].asInt();
                cout<<params[i]<<",";
            }

            _layers[i]->init(params);
            //_layers[i] = new InputLayer();

            cout<<"type:"<<type<<endl;
            cout<<"name:"<<name<<endl;
            cout<<"bottomLayerName:"<<bottom_layer_name<<endl;



            cout<<endl;
        }
    }

    is.close();
}

}
