#ifndef DONG_NET_MODEL_HPP_
#define DONG_NET_MODEL_HPP_
#include "common.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "layer.hpp"

using namespace std;

namespace dong
{

class NetModel
{
public:
    explicit NetModel():_input_shape_num(0),_input_shape_channels(0),_input_shape_height(0),_input_shape_width(0) {};
    virtual ~NetModel() {};
    virtual void train();
    virtual void save_model(const char* filename);
    virtual void load_model(const char* filename);
    virtual void setUp();

private:
    virtual void fillDataForOnceTrain(float* datas, int size);
    boost::shared_ptr<Neuron[]> _inputNeurons;
    boost::shared_ptr<Data> _inputData;
    boost::shared_ptr<Layer[]> _layers;
    DISABLE_COPY_AND_ASSIGN(NetModel);
    int _input_shape_num;
    int _input_shape_channels;
    int _input_shape_height;
    int _input_shape_width;
    Layer* getLayersByName(const char* name);
};

}


#endif  // DONG_NET_MODEL_HPP_
