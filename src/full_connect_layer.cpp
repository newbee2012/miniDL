#include <iostream>
#include "full_connect_layer.hpp"
using namespace std;

namespace dong
{

void FullConnectLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    _top_data.reset(new Data(_bottom_data->num(), _output_count, 1, 1, CONSTANT));
    _bias_data.reset(new Data(_output_count, 1, 1, 1, CONSTANT));
    int _input_count = _bottom_data->channels()*_bottom_data->height()*_bottom_data->width();
    _weight_data.reset(new Data(_output_count, _input_count, 1, 1, XAVIER));
    for (int t_n = 0; t_n < _top_data->num(); ++t_n)
    {
        for (int t_c = 0; t_c < _output_count; ++t_c)
        {
            Neuron* t_neuron = _top_data->get(t_n, t_c, 0, 0);
            t_neuron->_bias = _bias_data->get(t_c);
            for (int b_c = 0; b_c < _bottom_data->channels(); ++b_c)
            {
                for (int b_h = 0; b_h < _bottom_data->height(); ++b_h)
                {
                    for (int b_w = 0; b_w < _bottom_data->width(); ++b_w)
                    {
                        Neuron* b_neuron = _bottom_data->get(t_n, b_c, b_h, b_w);
                        Neuron* w_neuron = _weight_data->get(t_c, _bottom_data->offset(0, b_c, b_h, b_w),0,0);
                        b_neuron->_forward_neuron.push_back(t_neuron);
                        b_neuron->_weight_neuron.push_back(w_neuron);
                    }
                }
            }
        }
    }
}

void FullConnectLayer::forward_cpu()
{
    Layer::forwardBase();
}

void FullConnectLayer::backward_cpu()
{
    Layer::backwardBase();
}

void FullConnectLayer::init(int (&params)[6])
{
    this->_output_count = params[0];
}

}
