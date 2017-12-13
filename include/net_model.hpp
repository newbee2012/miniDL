#ifndef DONG_NET_MODEL_HPP_
#define DONG_NET_MODEL_HPP_
#include "common.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "layer.hpp"
#include "input_layer.hpp"
using namespace std;

namespace dong
{

class NetModel
{
public:
    explicit NetModel(string& modelDefineFilePath):_input_shape_num(0),_input_shape_channels(0),_input_shape_height(0),
        _input_shape_width(0),_per_batch_train_count(0),_batch_count(0),model_define_file_path(modelDefineFilePath)
    {};
    virtual ~NetModel() {};
    virtual void run();
    virtual void train()=0;
    virtual void test()=0;
    virtual void load_model();
    virtual void save_model();
    virtual void outputBmp();
    virtual void outputTime();

protected:
    virtual void fillDataForOnceTrainForward(Neuron* datas, int size, boost::shared_array<int>& labels);
    virtual void setUpInputLayer();
    virtual void forward();
    virtual void backward();
    virtual void update();
    virtual Layer* generateLayerByClassName(const char* className);
    boost::shared_array<Neuron> _input_neurons;
    boost::shared_ptr<Data> _input_data;
    boost::shared_ptr<Layer> _input_layer;
    boost::shared_ptr<Layer> _loss_layer;

    int _input_shape_num;
    int _input_shape_channels;
    int _input_shape_height;
    int _input_shape_width;
    int _per_batch_train_count;
    int _batch_count;
    string model_define_file_path;
    string model_data_file_path_in;
    string model_data_file_path_out;

    Mode _mode;
    DISABLE_COPY_AND_ASSIGN(NetModel);
};

}


#endif  // DONG_NET_MODEL_HPP_
