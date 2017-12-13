#include <iostream>
#include "conv_layer.hpp"
#include "neuron.hpp"
using namespace std;

namespace dong
{

void ConvLayer::init(int (&params)[4])
{
    _num_output = params[0];
    _kernel_h = params[1];
    _kernel_w = params[2];
}

void ConvLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    _top_data.reset(new Data(_bottom_data->num(), _num_output, _bottom_data->height()-_kernel_h + 1, _bottom_data->width()-_kernel_w + 1,
                             Data::CONSTANT));
    _bias_data.reset(new Data(1, _num_output,  _top_data->height(), _top_data->width(), Data::CONSTANT));
    _weight_data.reset(new Data(_num_output, _bottom_data->channels(), _kernel_h, _kernel_w, Data::XAVIER));

    int k_n = _weight_data->num();
    int k_c = _weight_data->channels();
    int k_h = _weight_data->height();
    int k_w = _weight_data->width();

    for (int t_n = 0; t_n < _top_data->num(); ++t_n)
    {
        for (int t_c = 0; t_c < _top_data->channels(); ++t_c)
        {
            for (int t_h = 0; t_h < _top_data->height(); ++t_h)
            {
                for (int t_w = 0; t_w < _top_data->width(); ++t_w)
                {
                    Neuron* t_neuron = _top_data->get(t_n, t_c, t_h, t_w);
                    Neuron* bias_neuron = _bias_data->get(0, t_c, t_h, t_w);
                    t_neuron->_bias = bias_neuron;
                    for(int k_c = 0; k_c < _weight_data->channels(); ++k_c)
                    {
                        for (int offset_h = 0; offset_h < k_h; ++offset_h)
                        {
                            for (int offset_w = 0; offset_w < k_w; ++offset_w)
                            {
                                Neuron* w_neuron = _weight_data->get(t_c, k_c, offset_h, offset_w);
                                for (int b_n = 0; b_n < _bottom_data->num(); ++b_n)
                                {
                                    Neuron* b_neuron = _bottom_data->get(b_n, k_c, t_h + offset_h, t_w + offset_w);
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
