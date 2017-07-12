#ifndef DONG_LAYER_HPP_
#define DONG_LAYER_HPP_
#include "common.hpp"
#include "data.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>


namespace dong
{
enum LayerType_ {INPUT_LAYER, CONVOLUTION_LAYER, POOL_LAYER, FULL_CONNECT_LAYER, RELU_LAYER, SOFTMAX_LAYER};
enum ForwardComputeType_ {INNER_PRODUCT, MAX, RELU};
enum Mode {TRAIN, TEST};

typedef LayerType_ LayerType;
typedef ForwardComputeType_ ForwardComputeType;

class ThreadParam
{
public:
    void init(Data* bottom_data, int offset_start, int offset_end, int threadIndex)
    {
        _bottom_data = bottom_data;
        _offset_start = offset_start;
        _offset_end = offset_end;
        _threadIndex = threadIndex;
    };

    Data* _bottom_data;
    int _offset_start;
    int _offset_end;
    int _threadIndex;
    dong::LayerType layerType;
};

class Layer
{
public:

    const char* EnumNames[6] = {"INPUT_LAYER", "CONVOLUTION_LAYER", "POOL_LAYER", "FULL_CONNECT_LAYER", "RELU_LAYER", "SOFTMAX_LAYER"};

    Layer() {};
    virtual ~Layer() {};
    virtual void forward_cpu() = 0;
    virtual void forward();
    virtual void backward_cpu() = 0;
    virtual void backward();
    virtual LayerType getType() = 0;
    static float BASE_LEARNING_RATE;

    inline virtual void setUp(const boost::shared_ptr<Data>& data)
    {
        this->_bottom_data = data;
    }

    inline virtual boost::shared_ptr<Data> getBottomData()
    {
        return _bottom_data;
    }

    inline virtual boost::shared_ptr<Data> getTopData()
    {
        return _top_data;
    }

    inline virtual boost::shared_ptr<Data> getWeightData()
    {
        return _weight_data;
    }

    inline virtual boost::shared_ptr<Data> getBiasData()
    {
        return _bias_data;
    }

protected:
    virtual void forwardBase();
    virtual void backwardBase();
    static void* backwardBaseThread(void* ptr);
    static void backwardLimit(Data* _bottom_data, int offset_start, int offset_end);
    virtual void updateWeight();
    virtual void updateBias();
    boost::shared_ptr<Data> _bottom_data;
    boost::shared_ptr<Data> _top_data;
    boost::shared_ptr<Data> _weight_data;
    boost::shared_ptr<Data> _bias_data;

    ForwardComputeType _forwardType;

    DISABLE_COPY_AND_ASSIGN(Layer);
};

}  // namespace dong

#endif  // DONG_LAYER_HPP_
