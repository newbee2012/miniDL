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
void NetModel::run()
{
    cout<<"------------load model--------------"<<endl;
    load_model();
    if(_mode == TRAIN)
    {
        cout<<"------------train start--------------"<<endl;
        train();
        outputBmp();
        cout<<"------------save model--------------"<<endl;
        save_model();
    }
    else if(_mode == TEST)
    {
        cout<<"------------test start--------------"<<endl;
        test();
    }
}

void NetModel::outputBmp()
{
    boost::shared_ptr<Layer> layer = _input_layer;
    while(layer->getTopLayer().get())
    {
        layer = layer->getTopLayer();
        if(layer->getType() == CONVOLUTION_LAYER)
        {
            string filepath("/home/chendejia/workspace/github/miniDL/bin/Release/");
            filepath.append(layer->getName()).append("_weight");
            layer->getWeightData()->genBmp(filepath);

            filepath = "/home/chendejia/workspace/github/miniDL/bin/Release/";
            filepath.append(layer->getName()).append("_topdata");
            layer->getTopData()->genBmp(filepath);
        }
    }
}

void NetModel::forward()
{
    boost::shared_ptr<Layer> layer = _input_layer->getTopLayer();
    while(layer.get())
    {
        layer->forward();
        layer = layer->getTopLayer();
    }
}

void NetModel::backward()
{
    boost::shared_ptr<Layer> layer = _loss_layer;
    while(layer.get())
    {
        layer->backward();
        layer = layer->getBottomLayer();
    }
}

void NetModel::update()
{
    boost::shared_ptr<Layer> layer = _input_layer->getTopLayer();
    while(layer.get())
    {
        if(layer->getType() == CONVOLUTION_LAYER || layer->getType() == FULL_CONNECT_LAYER)
        {
            layer->updateWeight();
            layer->updateBias();
        }

        layer = layer->getTopLayer();
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
        return new SoftmaxLayer(_mode);
    }

    return NULL;
}

void NetModel::save_model()
{
    cout<<"saving model...."<<endl;
    Json::Reader reader;
    Json::Value root;

    std::ifstream is;
    is.open (this->model_define_file_path.c_str(), std::ios::binary );
    if (!reader.parse(is, root))
    {
        ASSERT(false, cout<<"Json 解析失败！"<<endl);
    }

    Json::Value jo_layers = root["layersModel"];
    Json::Value jo_layer = jo_layers["inputLayer"];
    boost::shared_ptr<Layer> layer = _input_layer;
    ASSERT(!jo_layer.isNull(), cout<<"inputLayer不存在！"<<endl);

    string layerName = jo_layer["topLayer"].asString();
    jo_layer = jo_layers[layerName];
    layer = layer->getTopLayer();

    Json::Value dataRoot;
    while(!jo_layer.isNull())
    {
        Json::Value weightArray;
        Json::Value biasArray;
        if(layer->getType() == CONVOLUTION_LAYER || layer->getType() == FULL_CONNECT_LAYER)
        {
            string genBmpBasePath = "/home/chendejia/workspace/github/miniDL/bin/Release/";
            genBmpBasePath.append(layerName);
            if(layer->getType() == CONVOLUTION_LAYER)
            {
                //genBmpBasePath += "_weight";
                //layer->getWeightData()->genBmp(genBmpBasePath.c_str());
            }else if(layer->getType() == FULL_CONNECT_LAYER)
            {
                genBmpBasePath += "_topdata";
                layer->getTopData()->genBmp(genBmpBasePath);
            }


            for(int i = 0; i< layer->getWeightData() ->count();++i)
            {
                Neuron* neuron = layer->getWeightData()->get(i);
                weightArray[i] = neuron->_value;
            }

            dataRoot[layerName]["weight"] = weightArray;

            boost::shared_ptr<Data> biasData = layer->getBiasData();
            for(int i = 0; i< biasData ->count();++i)
            {
                Neuron* neuron = biasData->get(i);
                biasArray[i] = neuron->_value;
            }

            dataRoot[layerName]["bias"] = biasArray;
        }

        layerName = jo_layer["topLayer"].asString();
        jo_layer = jo_layers[layerName];
        layer = layer->getTopLayer();
    }

    Json::StyledWriter writer;
    std::string strWrite = writer.write(dataRoot);
    std::ofstream ofs;
    cout<<this->model_data_file_path_in<<endl;
    ofs.open(this->model_data_file_path_out.c_str());
    ofs << strWrite;
    ofs.close();
}

void NetModel::load_model()
{
    Layer::BASE_LEARNING_RATE = 0.0002F;
    Layer::LEARNING_RATE_POLICY = INV;
    Layer::GAMMA = 0.0001F;
    Layer::MOMENTUM = 0.9F;
    Layer::POWER = 0.75F;
    Layer::WEIGHT_DECAY = 0.0005F;
    Layer::STEPSIZE = 100;
    Layer::CURRENT_ITER_COUNT = 0;
    Layer::BATCH_SIZE = 1;
    cout<<"loading model...."<<endl;

    Json::Reader reader;

    Json::Value modelDefineRoot;
    Json::Value modelDataRoot;
    std::ifstream is;
    is.open (this->model_define_file_path.c_str(), std::ios::binary );
    if (!reader.parse(is, modelDefineRoot))
    {
        ASSERT(false, cout<<this->model_define_file_path<< " 解析失败！"<<endl);
    }

    is.close();

    string modelDataFilePathIn = modelDefineRoot["modelDataFilePathIn"].asString();
    string modelDataFilePathOut = modelDefineRoot["modelDataFilePathOut"].asString();
    this->model_data_file_path_in = modelDataFilePathIn;
    this->model_data_file_path_out = modelDataFilePathOut;

    is.open (model_data_file_path_in.c_str(), std::ios::binary );
    if (!reader.parse(is, modelDataRoot))
    {
        ASSERT(false, cout<<this->model_data_file_path_in<< " 解析失败！"<<endl);
    }
    else
    {
        cout<<this->model_data_file_path_in<<endl;
        cout<<this->model_data_file_path_out<<endl;
    }

    is.close();

    Json::Value jo_mode = modelDefineRoot["mode"];
    _mode = STRING_TO_MODE(jo_mode.asString().c_str());
    ASSERT(_mode >= 0 && _mode < MODE_SIZE, cout<<"mode 未定义或取值非法！"<<endl);

    //读取超参数
    Json::Value jo_hyperParameters = modelDefineRoot["hyperParameters"];
    ASSERT(!jo_hyperParameters.isNull(), cout<<"节点hyperParameters不存在！"<<endl);

    _per_batch_train_count = jo_hyperParameters["perBatchTrainCount"].asInt();
    ASSERT(_per_batch_train_count>0, cout<<"perBatchTrainCount 未定义或取值非法！"<<endl);
    Layer::BATCH_SIZE = _per_batch_train_count;

    _batch_count = jo_hyperParameters["batchCount"].asInt();
    ASSERT(_batch_count>0, cout<<"batchCount 未定义或取值非法！"<<endl);

    Layer::BASE_LEARNING_RATE = jo_hyperParameters["BASE_LEARNING_RATE"].asFloat();
    ASSERT(Layer::BASE_LEARNING_RATE > 0, cout<<"Layer::BASE_LEARNING_RATE 未定义或取值非法！"<<endl);

    Layer::LEARNING_RATE_POLICY = STRING_TO_LR_POLICY(jo_hyperParameters["LEARNING_RATE_POLICY"].asString().c_str());
    ASSERT(Layer::LEARNING_RATE_POLICY >= 0 && Layer::LEARNING_RATE_POLICY < LR_Policy_size, cout<<"Layer::LEARNING_RATE_POLICY 未定义或取值非法！"<<endl);

    Layer::GAMMA = jo_hyperParameters["GAMMA"].asFloat();
    ASSERT(Layer::GAMMA > 0, cout<<"Layer::GAMMA 未定义或取值非法！"<<endl);

    Layer::MOMENTUM = jo_hyperParameters["MOMENTUM"].asFloat();
    ASSERT(Layer::MOMENTUM > 0, cout<<"Layer::MOMENTUM 未定义或取值非法！"<<endl);

    Layer::POWER = jo_hyperParameters["POWER"].asFloat();
    ASSERT(Layer::POWER > 0, cout<<"Layer::POWER 未定义或取值非法！"<<endl);

    Layer::WEIGHT_DECAY = jo_hyperParameters["WEIGHT_DECAY"].asFloat();
    ASSERT(Layer::WEIGHT_DECAY > 0, cout<<"Layer::WEIGHT_DECAY 未定义或取值非法！"<<endl);

    Layer::STEPSIZE = jo_hyperParameters["STEPSIZE"].asInt();
    ASSERT(Layer::STEPSIZE > 0, cout<<"Layer::STEPSIZE 未定义或取值非法！"<<endl);

    //读取输入数据尺寸
    Json::Value jo_input_shape = modelDefineRoot["inputShape"];
    ASSERT(!jo_input_shape.isNull(), cout<<"节点inputShape不存在！"<<endl);

    _input_shape_num = jo_input_shape["num"].asInt();
    _input_shape_channels = jo_input_shape["channels"].asInt();
    _input_shape_height = jo_input_shape["height"].asInt();
    _input_shape_width = jo_input_shape["width"].asInt();

    //读取layers定义
    Json::Value jo_layers = modelDefineRoot["layersModel"];
    cout<<"model json:"<<endl<<jo_layers<<endl;
    ASSERT(!jo_layers.isNull(), cout<<"节点layersModel不存在！"<<endl);

    //查找输入层
    string layerName = "inputLayer";

    Json::Value jo_layer = jo_layers[layerName];
    ASSERT(!jo_layer.isNull(), cout<<layerName<<" 未定义！"<<endl);

    boost::shared_ptr<Layer> bottom_layer;

    while(!jo_layer.isNull())
    {
        const string impl_class = jo_layer["implClass"].asString();
        Json::Value init_params = jo_layer["initParams"];
        int params_size = init_params.size();
        int params[4] = {0};
        for(int j = 0; j < params_size; ++j)
        {
            params[j] = init_params[j].asInt();
        }

        boost::shared_ptr<Layer> layer(generateLayerByClassName(impl_class.c_str()));
        layer->setName(layerName);
        layer->init(params);
        if(layer->getType() == INPUT_LAYER)
        {
            _input_layer = layer;
            this->setUpInputLayer();
        }
        else if(layer->getType() == LOSS_LAYER)
        {
            _loss_layer = layer;
        }

        if(NULL != bottom_layer)
        {
            bottom_layer->setTopLayer(layer);
            layer->setBottomLayer(bottom_layer);
        }

        if(layer->getType() == CONVOLUTION_LAYER || layer->getType() == FULL_CONNECT_LAYER)
        {
            Json::Value jo_weights = modelDataRoot[layerName]["weight"];
            Json::Value jo_bias = modelDataRoot[layerName]["bias"];
            boost::shared_ptr<Data>& weightData = layer->getWeightData();
            boost::shared_ptr<Data>& biasData = layer->getBiasData();
            ASSERT(jo_weights.size() == weightData->count() || jo_weights.size() == 0, cout<<layerName<<"读取weights个数和model定义不一致"<<endl);
            ASSERT(jo_bias.size() == biasData->count() || jo_bias.size() == 0, cout<<layerName<<"读取bias个数和model定义不一致"<<endl);

            for(int i = 0; i < jo_weights.size(); ++i)
            {
                Neuron* neuron = weightData->get(i);
                neuron->_value = jo_weights[i].asFloat();
            }

            for(int i = 0; i < jo_bias.size(); ++i)
            {
                Neuron* neuron = biasData->get(i);
                neuron->_value = jo_bias[i].asFloat();
            }

            //layer->getWeightData()->print();
        }

        bottom_layer = layer;

        layerName = jo_layer["topLayer"].asString();
        jo_layer = jo_layers[layerName];
    }


}

}
