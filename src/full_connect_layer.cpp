#include <iostream>
#include "full_connect_layer.hpp"
using namespace std;

namespace dong
{

void FullConnectLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    _weight_data.reset(new Data(_num, _bottom_data->count(), 1, 1, Data::XAVIER));
    int t_n = 1;
    int t_h = 1;
    int t_w = _num;
    _top_data.reset(new Data(t_n, 1, t_h, t_w, Data::CONSTANT));
    _bias_data.reset(new Data(t_n, 1, t_h, t_w, Data::CONSTANT));

    for (int n = 0; n < t_n; n++) {
        for (int h = 0; h < t_h; h++) {
            for (int w = 0; w < t_w; w++) {
                Neuron* t_neuron = _top_data->get(n, 0, h, w);
                Neuron* bias_neuron = _bias_data->get(n, 0, h, w);
                t_neuron->_bias = bias_neuron;

                for (int i = 0; i < _bottom_data->count(); i++) {
                    Neuron* b_neuron = _bottom_data->get(i);
                    Neuron* w_neuron = _weight_data->get(w, i, 0, 0);
                    b_neuron->_forward_neuron.push_back(t_neuron);
                    b_neuron->_weight_neuron.push_back(w_neuron);
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
}

}
