#include <iostream>
#include "conv_layer.hpp"
#include "neuron.hpp"
#include "common.hpp"
using namespace std;

namespace dong
{

void ConvLayer::init(int (&params)[6])
{
    _num_output = params[0];
    _kernel_h = params[1];
    _kernel_w = params[2];
    _pad_h = params[3];
    _pad_w = params[4];
    _stride = params[5];
}

void ConvLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);

    int top_height=(_bottom_data->height() + 2 * _pad_h - _kernel_h) /_stride + 1;
    int top_width=(_bottom_data->width() + 2 * _pad_w - _kernel_w) /_stride + 1;

    _top_data.reset(new Data(_bottom_data->num(), _num_output, top_height, top_width, CONSTANT));
    _bias_data.reset(new Data(_num_output,  _top_data->height(), _top_data->width(), 1, CONSTANT));
    _weight_data.reset(new Data(_num_output, _bottom_data->channels(), _kernel_h, _kernel_w, XAVIER));

    for (int t_n = 0; t_n < _top_data->num(); ++t_n)
    {
        for (int t_c = 0; t_c < _top_data->channels(); ++t_c)
        {
            for (int t_h = 0; t_h < _top_data->height(); ++t_h)
            {
                int b_h_start = t_h * _stride - _pad_h;
                for (int t_w = 0; t_w < _top_data->width(); ++t_w)
                {
                    int b_w_start = t_w * _stride - _pad_w;
                    Neuron* t_neuron = _top_data->get(t_n, t_c, t_h, t_w);
                    Neuron* bias_neuron = _bias_data->get(t_c, t_h, t_w, 0);
                    t_neuron->_bias = bias_neuron;
                    for(int k_c = 0; k_c < _weight_data->channels(); ++k_c)
                    {
                        for (int offset_h = 0; offset_h < _weight_data->height(); ++offset_h)
                        {
                            for (int offset_w = 0; offset_w < _weight_data->width(); ++offset_w)
                            {
                                int b_h= b_h_start + offset_h;
                                int b_w= b_w_start + offset_w;
                                if(b_h >=0 && b_h < _bottom_data->height() && b_w >=0 && b_w < _bottom_data->width())
                                {
                                    Neuron* w_neuron = _weight_data->get(t_c, k_c, offset_h, offset_w);
                                    Neuron* b_neuron = _bottom_data->get(t_n, k_c, b_h_start + offset_h, b_w_start + offset_w);
                                    b_neuron->_forward_neuron.push_back(t_neuron);
                                    b_neuron->_weight_neuron.push_back(w_neuron);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}

void ConvLayer::forward_cpu()
{
    Layer::forwardBase();
}

void ConvLayer::backward_cpu()
{
    Layer::backwardBase();
}



}
