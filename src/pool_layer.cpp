#include <iostream>
#include "pool_layer.hpp"
using namespace std;

namespace dong
{

void PoolLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    int b_n = _bottom_data->num();
    int b_h = _bottom_data->height();
    int b_w = _bottom_data->width();
    int t_n = b_n;
    int t_h = (b_h - _kernel_h) / _stride_h + 1;
    int t_w = (b_w - _kernel_w) / _stride_w + 1;
    _top_data.reset(new Data(t_n, 1, t_h, t_w, Data::CONSTANT));
    _weight_data.reset(new Data(t_n, 1, t_h * _kernel_h, t_w * _kernel_w, Data::CONSTANT));
    for (int n = 0; n < t_n; n++) {
        for (int h = 0; h < t_h; h++) {
            for (int w = 0; w < t_w; w++) {
                Neuron* t_neuron = _top_data->get(n, 0, h, w);
                for (int offset_h = 0; offset_h < _kernel_h; offset_h++) {
                    for (int offset_w = 0; offset_w < _kernel_w; offset_w++) {
                        Neuron* b_neuron = _bottom_data->get(n, 0, h * _stride_h + offset_h, w * _stride_w + offset_w);
                        Neuron* w_neuron = _weight_data->get(n, 0, h * _kernel_h + offset_h, w * _kernel_w + offset_w);
                        b_neuron->_forward_neuron.push_back(t_neuron);
                        b_neuron->_weight_neuron.push_back(w_neuron);
                    }
                }
            }
        }
    }
}

void PoolLayer::forward_cpu()
{
    int t_n = _top_data->num();
    int t_h = _top_data->height();
    int t_w = _top_data->width();
    for (int n = 0; n < t_n; n++) {
        for (int h = 0; h < t_h; h++) {
            for (int w = 0; w < t_w; w++) {
                Neuron* t_neuron = _top_data->get(n, 0, h, w);
                int max_index = _weight_data->offset(n, 0, h * _kernel_h + 0, w * _kernel_w + 0);
                float max_value = _bottom_data->get(n, 0, h * _stride_h + 0, w * _stride_w + 0)->_value;
                for (int offset_h = 0; offset_h < _kernel_h; offset_h++) {
                    for (int offset_w = 0; offset_w < _kernel_w; offset_w++) {
                        Neuron* b_neuron = _bottom_data->get(n, 0, h * _stride_h + offset_h, w * _stride_w + offset_w);
                        Neuron* w_neuron = _weight_data->get(n, 0, h * _kernel_h + offset_h, w * _kernel_w + offset_w);
                        if (b_neuron->_value > max_value) {
                            max_value = b_neuron->_value;
                            max_index = _weight_data->offset(n, 0, h * _kernel_h + offset_h, w * _kernel_w + offset_w);
                        }
                    }
                }

                t_neuron->_value = max_value;
                _weight_data->get(max_index)->_value = 1.0F;
            }
        }
    }
}

void PoolLayer::backward_cpu()
{
    Layer::backwardBase();
}

void PoolLayer::init(int kernel_h, int kernel_w, int stride_h, int stride_w)
{
    this->_kernel_h = kernel_h;
    this->_kernel_w = kernel_w;
    this->_stride_h = stride_h;
    this->_stride_w = stride_w;
}

}
