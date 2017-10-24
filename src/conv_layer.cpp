#include <iostream>
#include "conv_layer.hpp"
#include "neuron.hpp"
using namespace std;

namespace dong
{

void ConvLayer::init(int num_output, int kernel_h, int kernel_w)
{
    _num_output = num_output;
    _kernel_h = kernel_h;
    _kernel_w = kernel_w;
}

void ConvLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    int b_n = _bottom_data->num();
    int b_h = _bottom_data->height();
    int b_w = _bottom_data->width();
    _weight_data.reset(new Data(_num_output, b_n, _kernel_h, _kernel_w, Data::XAVIER));
    int k_n = _weight_data->num();
    int k_h = _weight_data->height();
    int k_w = _weight_data->width();
    int t_n = k_n;
    int t_h = b_h - k_h + 1;
    int t_w = b_w - k_w + 1;
    _top_data.reset(new Data(_num_output, 1, t_h, t_w, Data::CONSTANT));
    _bias_data.reset(new Data(_num_output, 1, t_h, t_w, Data::CONSTANT));

    for (int n = 0; n < t_n; n++) {
        for (int h = 0; h < t_h; h++) {
            for (int w = 0; w < t_w; w++) {
                Neuron* t_neuron = _top_data->get(n, 0, h, w);
                Neuron* bias_neuron = _bias_data->get(n, 0, h, w);
                t_neuron->_bias = bias_neuron;

                for (int c = 0; c < b_n; c++) {
                    for (int offset_h = 0; offset_h < k_h; offset_h++) {
                        for (int offset_w = 0; offset_w < k_w; offset_w++) {
                            Neuron* b_neuron = _bottom_data->get(c, 0, h + offset_h, w + offset_w);
                            Neuron* w_neuron = _weight_data->get(n, c, offset_h, offset_w);
                            b_neuron->_forward_neuron.push_back(t_neuron);
                            b_neuron->_weight_neuron.push_back(w_neuron);
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
