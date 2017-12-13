#include <iostream>
#include "full_connect_layer.hpp"
using namespace std;

namespace dong
{

void FullConnectLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    int t_n = _num;
    int t_c = _channels;
    int t_h = _height;
    int t_w = _width;
    int t_count = _num * _channels *_height * _width;


    _top_data.reset(new Data(_num, _channels, _height, _width, Data::CONSTANT));
    _weight_data.reset(new Data(t_count, _bottom_data->channels(), _bottom_data->height(), _bottom_data->width(), Data::XAVIER));
    _bias_data.reset(new Data(t_n, t_c, t_h, t_w, Data::CONSTANT));

    for (int n = 0; n < t_n; n++)
    {
        for (int c = 0; c < t_c; c++)
        {
            for (int h = 0; h < t_h; h++)
            {
                for (int w = 0; w < t_w; w++)
                {
                    Neuron* t_neuron = _top_data->get(n, c, h, w);
                    Neuron* bias_neuron = _bias_data->get(n, c, h, w);
                    t_neuron->_bias = bias_neuron;
                    for (int i = 0; i < _bottom_data->count(); i++)
                    {
                        Neuron* b_neuron = _bottom_data->get(i);
                        Neuron* w_neuron = _weight_data->get(_top_data->offset(n, c, h, w), i, 0, 0);
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

void FullConnectLayer::init(int (&params)[4])
{
    this->_num = params[0];
    this->_channels = params[1];
    this->_height = params[2];
    this->_width = params[3];
}

}
