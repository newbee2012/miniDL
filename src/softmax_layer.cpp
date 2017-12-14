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
    _num = _bottom_data->num();
    _forecast_labels.reset(new int[_num]{-1});
    int input_count = _bottom_data->channels() *_bottom_data->height() * _bottom_data->width();
    _top_data.reset(new Data(_num, input_count, 1, 1, Data::CONSTANT));
}

void SoftmaxLayer::forward_cpu()
{
    _loss = 0.0F;
    int input_count = _bottom_data->channels() *_bottom_data->height() * _bottom_data->width();
    for(int n = 0; n < _top_data->num(); ++n)
    {
        float maxValue = -FLT_MAX;
        for (int c = 0; c < input_count; ++c)
        {
            if(_bottom_data->get(n, c, 0, 0)->_value > maxValue)
            {
                _forecast_labels[n] = c;
                maxValue = _bottom_data->get(n, c, 0, 0)->_value;
            }
        }

        double sumExp = 0.0F;

        for (int c = 0; c < input_count; ++c)
        {
            float expValue = exp(_bottom_data->get(n, c, 0, 0)->_value - maxValue) ;
            _top_data->get(n,c,0,0)->_value = expValue;
            sumExp += expValue;
        }

        for (int c = 0; c < input_count; ++c)
        {
            _top_data->get(n,c,0,0)->_value = (float)_top_data->get(n,c,0,0)->_value / sumExp;
        }

        _loss += -log(std::max(_top_data->get(n, _labels[n], 0, 0)->_value, FLT_MIN));
    }

    _loss /= _top_data->num();
}

void SoftmaxLayer::backward_cpu()
{
    for (int n = 0; n < _top_data->num(); ++n)
    {
        for (int c = 0; c < _top_data->channels(); ++c)
        {
            Neuron* b_neuron = _bottom_data->get(n, c, 0, 0);
            Neuron* t_neuron = _top_data->get(n, c, 0, 0);
            b_neuron->_diff = t_neuron->_value;
            if (NULL != b_neuron->_bias)
            {
                b_neuron->_bias->_diff = b_neuron->_diff;
            }
        }


        Neuron* label_neuron = _bottom_data->get(n, _labels[n], 0, 0);
        label_neuron->_diff -= 1;

        if (NULL != label_neuron->_bias)
        {
            label_neuron->_bias->_diff -= label_neuron->_diff;
        }
    }

    for (int i = 0; i < _top_data->count(); ++i)
    {
        Neuron* b_neuron = _bottom_data->get(i);
        b_neuron->_diff /= _top_data->num();
        if (NULL != b_neuron->_bias)
        {
            b_neuron->_bias->_diff /= _top_data->num();
        }
    }
}

void SoftmaxLayer::setLabels(boost::shared_array<int>& labels)
{
    _labels = labels;
}

void SoftmaxLayer::init(int (&params)[4])
{
}

}
