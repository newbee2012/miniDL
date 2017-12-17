#include <iostream>
#include <pthread.h>
#include "layer.hpp"

#define RECORD_FORWARD_TIME
#define RECORD_BACKWARD_TIME
using namespace std;
namespace dong
{

boost::shared_ptr<Layer>& Layer::getTopLayer()
{
    return this->_top_layer;
}


void Layer::setTopLayer(boost::shared_ptr<Layer>& layer)
{
    _top_layer = layer;
    _top_layer->setUp(this->getTopData());
}

boost::shared_ptr<Layer>& Layer::getBottomLayer()
{
    return this->_bottom_layer;
}


void Layer::setBottomLayer(boost::shared_ptr<Layer>& layer)
{
    _bottom_layer = layer;
}


void Layer::forward()
{
#ifdef RECORD_FORWARD_TIME
    boost::posix_time::ptime start_cpu_;
    boost::posix_time::ptime stop_cpu_;
    start_cpu_ = boost::posix_time::microsec_clock::local_time();
#endif

    _bottom_data->clearDiff();
    _top_data->clearValue();
    if (_weight_data.get() != NULL)
    {
        _weight_data->clearDiff();

        switch (getType())
        {
        case POOL_LAYER:
        case RELU_LAYER:
        case LOSS_LAYER:
            _weight_data->clearValue();
            break;

        default:
            break;
        }
    }

    if (_bias_data != NULL)
    {
        _bias_data->clearDiff();
    }

    this->forward_cpu();

#ifdef RECORD_FORWARD_TIME
    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    _sum_forward_time += (stop_cpu_ - start_cpu_).total_microseconds();
#endif
}

void Layer::backward()
{
#ifdef RECORD_BACKWARD_TIME
    boost::posix_time::ptime start_cpu_;
    boost::posix_time::ptime stop_cpu_;
    start_cpu_ = boost::posix_time::microsec_clock::local_time();
#endif

    this->backward_cpu();

#ifdef RECORD_BACKWARD_TIME
    stop_cpu_ = boost::posix_time::microsec_clock::local_time();
    _sum_backward_time += (stop_cpu_ - start_cpu_).total_microseconds();
#endif
}

void Layer::forwardBase()
{
    int thread_count = Layer::FORWARD_THREAD_COUNT;
    if (thread_count > 1)
    {
        int spilit_num = _bottom_data->num() / thread_count;
        if(spilit_num < 1)
        {
            spilit_num = 1;
            thread_count = _bottom_data->num();
        }

        pthread_t thread[thread_count];
        ThreadParam p[thread_count];

        int count = _bottom_data->offset(1,0,0,0);
        for (int i = 0; i < thread_count; ++i)
        {
            int offset_num_start = i * spilit_num;
            int offset_num_end = offset_num_start + spilit_num;

            if (i == thread_count - 1)
            {
                offset_num_end = _bottom_data->num();
            }

            p[i].init(_bottom_data.get(), offset_num_start * count, offset_num_end * count, i);
            pthread_create(&thread[i], NULL, forwardBaseThread, &p[i]);
        }

        for (int i = 0; i < thread_count; ++i)
        {
            pthread_join(thread[i], NULL);
        }
    }
    else
    {
        Layer::forwardLimit(_bottom_data.get(), 0, _bottom_data->count());
    }

    if (NULL != _bias_data.get())
    {
        for (int n = 0; n < _top_data->num(); ++n)
        {
            for (int c = 0; c < _top_data->channels(); ++c)
            {
                for (int h = 0; h < _top_data->height(); ++h)
                {
                    for (int w = 0; w < _top_data->width(); ++w)
                    {
                        _top_data->get(n,c,h,w)->_value += _bias_data->get(c,h,w,0)->_value;
                    }
                }
            }
        }
    }
}

void Layer::backwardBase()
{
    int thread_count = Layer::BACKWARD_THREAD_COUNT;
    if (thread_count > 1)
    {
        int spilit_count = _bottom_data->count() / thread_count;
        if(spilit_count < 1)
        {
            spilit_count = 1;
            thread_count = _bottom_data->count();
        }

        pthread_t thread[thread_count];
        ThreadParam p[thread_count];
        for (int i = 0; i < thread_count; ++i)
        {
            int offset_start = i * spilit_count;
            int offset_end = offset_start + spilit_count;

            if (i == thread_count - 1)
            {
                offset_end = _bottom_data->count();
            }

            p[i].init(_bottom_data.get(), offset_start, offset_end, i);
            pthread_create(&thread[i], NULL, backwardBaseThread, &p[i]);
        }

        for (int i = 0; i < thread_count; ++i)
        {
            pthread_join(thread[i], NULL);
        }
    }
    else
    {
        Layer::backwardLimit(_bottom_data.get(), 0, _bottom_data->count());
    }
};

void* Layer::forwardBaseThread(void* ptr)
{
    ThreadParam* p = (ThreadParam*)ptr;
    Data* bottom_data = p->_bottom_data;
    Layer::forwardLimit(bottom_data, p->_offset_start, p->_offset_end);
    return 0;
}

void Layer::forwardLimit(Data* bottom_data, int offset_start, int offset_end)
{
    for (int i = offset_start; i < offset_end; ++i)
    {
        bottom_data->get(i)->forward();
    }
}

void Layer::backwardLimit(Data* bottom_data, int offset_start, int offset_end)
{
    for (int i = offset_start; i < offset_end; ++i)
    {
        bottom_data->get(i)->backward();
    }
}

void* Layer::backwardBaseThread(void* ptr)
{
    ThreadParam* p = (ThreadParam*)ptr;
    Data* bottom_data = p->_bottom_data;
    Layer::backwardLimit(bottom_data, p->_offset_start, p->_offset_end);
    return 0;
}

void Layer::updateWeight()
{
    for (int i = 0; i < _weight_data->count(); ++i)
    {
        Neuron* w_neuron = _weight_data->get(i);
        float diff =  w_neuron->_diff;
        diff += w_neuron->_value * Layer::WEIGHT_DECAY;
        diff *= CURRENT_LEARNING_RATE * _lr_mult_weight;
        diff += Layer::MOMENTUM * w_neuron->_history_diff;
        w_neuron->_history_diff = diff;
        w_neuron->_value -= diff;
    }
}

void Layer::updateBias()
{
    for (int i = 0; i < _bias_data->count(); ++i)
    {
        Neuron* bias_neuron = _bias_data->get(i);
        float diff = bias_neuron->_diff;
        diff += bias_neuron->_value * Layer::WEIGHT_DECAY;
        diff *= CURRENT_LEARNING_RATE * _lr_mult_bias;
        diff += Layer::MOMENTUM * bias_neuron->_history_diff;
        bias_neuron->_history_diff = diff;
        bias_neuron->_value -= diff;
    }
}

float Layer::getLearningRate()
{
    switch (Layer::LEARNING_RATE_POLICY)
    {
    case FIXED:
        return BASE_LEARNING_RATE;

    case STEP:
        return BASE_LEARNING_RATE * std::pow(GAMMA, CURRENT_ITER_COUNT / STEPSIZE);

    case INV:
        return BASE_LEARNING_RATE * std::pow(1.0F + GAMMA * CURRENT_ITER_COUNT, -POWER);
    }
}
}
