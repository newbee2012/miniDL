#include <iostream>
#include "pool_layer.hpp"
using namespace std;

namespace dong
{

void PoolLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    int b_n = _bottom_data->num();
    int b_c = _bottom_data->channels();
    int b_h = _bottom_data->height();
    int b_w = _bottom_data->width();
    int t_n = b_n;
    int t_c = b_c;
    int t_h = (b_h - _kernel_h) / _stride_h + 1;
    int t_w = (b_w - _kernel_w) / _stride_w + 1;
    _top_data.reset(new Data(t_n, t_c, t_h, t_w, Data::CONSTANT));
    _weight_data.reset(new Data(t_n, t_c, t_h * _kernel_h, t_w * _kernel_w, Data::CONSTANT));
    for (int n = 0; n < t_n; ++n) {
        for (int c = 0; c < t_c; ++c) {
            for (int h = 0; h < t_h; ++h) {
                for (int w = 0; w < t_w; ++w) {
                    Neuron* t_neuron = _top_data->get(n, c, h, w);
                    for (int offset_h = 0; offset_h < _kernel_h; offset_h++) {
                        for (int offset_w = 0; offset_w < _kernel_w; offset_w++) {
                            Neuron* b_neuron = _bottom_data->get(n, c, h * _stride_h + offset_h, w * _stride_w + offset_w);
                            Neuron* w_neuron = _weight_data->get(n, c, h * _kernel_h + offset_h, w * _kernel_w + offset_w);
                            b_neuron->_forward_neuron.push_back(t_neuron);
                            b_neuron->_weight_neuron.push_back(w_neuron);
                        }
                    }
                }
            }
        }
    }
}

void PoolLayer::forward_cpu()
{
    int t_n = _top_data->num();
    int t_c = _top_data->channels();
    int t_h = _top_data->height();
    int t_w = _top_data->width();
    for (int n = 0; n < t_n; n++) {
        for (int c = 0; c < t_c; ++c) {
            for (int h = 0; h < t_h; h++) {
                for (int w = 0; w < t_w; w++) {
                    Neuron* t_neuron = _top_data->get(n, c, h, w);
                    int max_index = _weight_data->offset(n, c, h * _kernel_h + 0, w * _kernel_w + 0);
                    float max_value = _bottom_data->get(n, c, h * _stride_h + 0, w * _stride_w + 0)->_value;
                    for (int offset_h = 0; offset_h < _kernel_h; offset_h++) {
                        for (int offset_w = 0; offset_w < _kernel_w; offset_w++) {
                            Neuron* b_neuron = _bottom_data->get(n, c, h * _stride_h + offset_h, w * _stride_w + offset_w);
                            if (b_neuron->_value > max_value) {
                                max_value = b_neuron->_value;
                                max_index = _weight_data->offset(n, c, h * _kernel_h + offset_h, w * _kernel_w + offset_w);
                            }
                        }
                    }

                    t_neuron->_value = max_value;
                    _weight_data->get(max_index)->_value = 1.0F;
                }
            }
        }
    }
}

void PoolLayer::backward_cpu()
{
    Layer::backwardBase();
}

void PoolLayer::init(int (&params)[4])
{
    this->_kernel_h = params[0];
    this->_kernel_w = params[1];
    this->_stride_h = params[2];
    this->_stride_w = params[3];
}

}
