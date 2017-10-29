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
        Json::Value layers = root["layersModel"];
        cout<<"model json:"<<endl<<layers<<endl;
        if (!layers.isNull())  // 访问节点，Access an object value by name, create a null member if it does not exist.
        {
            int layers_size = layers.size();
            cout<<"layersSize:"<<layers_size<<endl;
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
                case INPUT_LAYER:
                    _layers[i] = new InputLayer();
                    break;
                case CONVOLUTION_LAYER:
                    _layers[i] = new ConvLayer();
                    break;
                default:
                    ASSERT(false, cout<<"不存在的LayerType-->"<<type.c_str());
                    break;
                }

                Json::Value init_params = layer["initParams"];
                int params_size = init_params.size();
                int* params = new int[params_size] ;
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
    }
    is.close();
}

}
