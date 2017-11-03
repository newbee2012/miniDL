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
    explicit NetModel():_input_shape_num(0),_input_shape_channels(0),_input_shape_height(0),
                        _input_shape_width(0),_per_iter_train_count(0),_batch_count(0) {};
    virtual ~NetModel() {};
    virtual void train()=0;
    virtual void save_model(const char* filename);
    virtual void load_model(const char* filename);
    virtual void setUpInputLayer();

protected:
    virtual void fillDataForOnceTrainForward(Neuron* datas, int size);
    virtual void forward();
    boost::shared_array<Neuron> _inputNeurons;
    boost::shared_ptr<Data> _inputData;
    boost::shared_ptr<Layer> _inputLayer;
    vector< boost::shared_ptr<Layer> > _layers;

    int _input_shape_num;
    int _input_shape_channels;
    int _input_shape_height;
    int _input_shape_width;
    int _per_iter_train_count;
    int _batch_count;

    DISABLE_COPY_AND_ASSIGN(NetModel);
};

}


#endif  // DONG_NET_MODEL_HPP_
