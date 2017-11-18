#include <iostream>
#include <common.hpp>
#include <algorithm>
#include "softmax_layer.hpp"

using namespace std;

namespace dong
{

void SoftmaxLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    _top_data.reset(new Data(1, 1, _bottom_data->height(), _bottom_data->width(), Data::CONSTANT));
}

void SoftmaxLayer::forward_cpu()
{
    _forecast_success = false;
    const int count = _bottom_data->count();
    float maxValue = 0.0F;

    int forecastLabel = 0;
    for (int i = 0; i < count; ++i)
    {
        if(_bottom_data->get(i)->_value > maxValue)
        {
            forecastLabel = i;
            maxValue = _bottom_data->get(i)->_value;
        }
    }

    if(_label == forecastLabel)
    {
        _forecast_success = true;
    }

    double sumExp = 0.0F;

    for (int i = 0; i < count; ++i)
    {
        float expValue = exp(_bottom_data->get(i)->_value - maxValue) ;
        _top_data->get(i)->_value = expValue;
        sumExp += expValue;
    }

    for (int i = 0; i < count; ++i)
    {
        _top_data->get(i)->_value = (float)_top_data->get(i)->_value / sumExp;
    }

    _loss = -log(std::max(_top_data->get(_label)->_value, FLT_MIN));

}

void SoftmaxLayer::backward_cpu()
{
    for (int i = 0; i < _bottom_data->count(); ++i)
    {
        Neuron* b_neuron = _bottom_data->get(i);
        Neuron* t_neuron = _top_data->get(i);
        b_neuron->_diff = t_neuron->_value;

        if (NULL != b_neuron->_bias)
        {
            b_neuron->_bias->_diff = b_neuron->_diff;
        }
    }

    _bottom_data->get(_label)->_diff -= 1;

    if (NULL != _bottom_data->get(_label)->_bias)
    {
        _bottom_data->get(_label)->_bias->_diff -= _bottom_data->get(_label)->_diff;
    }
}

void SoftmaxLayer::setLabel(int label)
{
    _label = label;
}

void SoftmaxLayer::init(int (&params)[4])
{
}

}
