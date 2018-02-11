#include <iostream>
#include "input_layer.hpp"
using namespace std;

namespace dong
{

void InputLayer::init(int (&params)[6])
{
}

void InputLayer::setUp(const boost::shared_ptr<Data>& data)
{
    Layer::setUp(data);
    _top_data.reset(new Data(_bottom_data->num(), _bottom_data->channels(), _bottom_data->height(), _bottom_data->width(), CONSTANT));
}

void InputLayer::forward_cpu()
{

    int count = _top_data->count();
    for(int i=0; i < count; ++i)
    {
        _top_data->get(i)->_value = _bottom_data->get(i)->_value * this->_scale;
    }
}

void InputLayer::backward_cpu()
{
}


}
