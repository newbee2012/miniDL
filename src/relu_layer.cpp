#include <iostream>
#include <common.hpp>
#include <algorithm>
#include "relu_layer.hpp"
using namespace std;

namespace dong
{

void ReluLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    _top_data.reset(new Data(_bottom_data->num(), _bottom_data->channels(), _bottom_data->height(), _bottom_data->width(), Data::CONSTANT));
    _weight_data.reset(new Data(_bottom_data->num(), _bottom_data->channels(), _bottom_data->height(), _bottom_data->width(), Data::CONSTANT));

    for (int i = 0; i < _bottom_data->count(); ++i) {
        Neuron* b_neuron = _bottom_data->get(i);
        Neuron* t_neuron = _top_data->get(i);
        Neuron* w_neuron = _weight_data->get(i);
        b_neuron->_forward_neuron.push_back(t_neuron);
        b_neuron->_weight_neuron.push_back(w_neuron);
    }
}

void ReluLayer::forward_cpu()
{
    for (int i = 0; i < _bottom_data->count(); ++i) {
        Neuron* b_neuron = _bottom_data->get(i);
        Neuron* t_neuron = _top_data->get(i);
        Neuron* w_neuron = _weight_data->get(i);

        if (b_neuron->_value > 0) {
            w_neuron->_value = 1.0F;
            t_neuron->_value = b_neuron->_value;
        } else {
            w_neuron->_value = 0.0F;
            t_neuron->_value = 0.0F;
        }
    }
}

void ReluLayer::backward_cpu()
{
    Layer::backwardBase();
}

void ReluLayer::init(int (&params)[4])
{
}
}
