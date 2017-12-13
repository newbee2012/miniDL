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
    if (_weight_data.get() != NULL) {
        _weight_data->clearDiff();

        switch (getType()) {
        case POOL_LAYER:
        case RELU_LAYER:
        case LOSS_LAYER:
            _weight_data->clearValue();
            break;

        default:
            break;
        }
    }

    if (_bias_data != NULL) {
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
    int bottom_count = _bottom_data->count();
    for (int i = 0; i < bottom_count; ++i) {
        _bottom_data->get(i)->forward();
    }

    if (NULL != _bias_data.get()) {
        int top_count = _top_data->count();
        for (int i = 0; i < top_count; ++i) {
            _top_data->get(i)->_value += _bias_data->get(i)->_value;
        }
    }
}

void Layer::backwardBase()
{
#define THREAD_COUNT 4
    bool multithreading = false;

    if (multithreading) {
        pthread_t thread[THREAD_COUNT];
        ThreadParam p[THREAD_COUNT];
        int spilit_count = _bottom_data->count() / THREAD_COUNT;

        for (int i = 0; i < THREAD_COUNT; ++i) {
            int offset_start = i * spilit_count;
            int offset_end = offset_start + spilit_count;

            if (i == THREAD_COUNT - 1) {
                offset_end = _bottom_data->count();
            }

            p[i].init(_bottom_data.get(), offset_start, offset_end, i);
            pthread_create(&thread[i], NULL, backwardBaseThread, &p[i]);
        }

        for (int i = 0; i < THREAD_COUNT; ++i) {
            pthread_join(thread[i], NULL);
        }
    } else {
        Layer::backwardLimit(_bottom_data.get(), 0, _bottom_data->count());
    }

    /*
    switch (getType()) {
    case CONVOLUTION_LAYER:
    case FULL_CONNECT_LAYER:
        updateWeight();
        updateBias();
        break;
    default:
        break;
    }*/
};


void Layer::backwardLimit(Data* bottom_data, int offset_start, int offset_end)
{
    for (int i = offset_start; i < offset_end; ++i) {
        bottom_data->get(i)->backward();
    }
}

void Layer::updateWeight()
{
    for (int i = 0; i < _weight_data->count(); ++i) {
        Neuron* w_neuron = _weight_data->get(i);
        float diff =  w_neuron->_batch_diff / Layer::BATCH_SIZE;
        diff += w_neuron->_value * Layer::WEIGHT_DECAY;
        diff *= CURRENT_LEARNING_RATE * _lr_mult_weight;
        diff += Layer::MOMENTUM * w_neuron->_history_diff;
        w_neuron->_history_diff = diff;
        w_neuron->_value -= diff;
        w_neuron->_batch_diff = 0.0F;
    }
}

void Layer::updateBias()
{
    for (int i = 0; i < _bias_data->count(); ++i) {
        Neuron* bias_neuron = _bias_data->get(i);
        float diff = bias_neuron->_batch_diff / Layer::BATCH_SIZE;
        diff += bias_neuron->_value * Layer::WEIGHT_DECAY;
        diff *= CURRENT_LEARNING_RATE * _lr_mult_bias;
        diff += Layer::MOMENTUM * bias_neuron->_history_diff;
        bias_neuron->_history_diff = diff;
        bias_neuron->_value -= diff;
        bias_neuron->_batch_diff = 0.0F;
    }
}

void* Layer::backwardBaseThread(void* ptr)
{
    ThreadParam* p = (ThreadParam*)ptr;
    Data* bottom_data = p->_bottom_data;
    Layer::backwardLimit(bottom_data, p->_offset_start, p->_offset_end);
    return 0;
}

float Layer::getLearningRate()
{
    switch (Layer::LEARNING_RATE_POLICY) {
    case FIXED:
        return BASE_LEARNING_RATE;

    case STEP:
        return BASE_LEARNING_RATE * std::pow(GAMMA, CURRENT_ITER_COUNT / STEPSIZE);

    case INV:
        return BASE_LEARNING_RATE * std::pow(1.0F + GAMMA * CURRENT_ITER_COUNT, -POWER);
    }
}
}
