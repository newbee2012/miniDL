#ifndef DONG_SOFTMAX_LAYER_HPP_
#define DONG_SOFTMAX_LAYER_HPP_
#include "common.hpp"
#include "layer.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>


namespace dong
{
class LossLayer: public Layer
{
public:
    virtual float getLoss() = 0;
    virtual bool getForecastResult() = 0;
};

class SoftmaxLayer: public LossLayer
{
public:

    explicit SoftmaxLayer(Mode mode):_mode(mode), _forecast_success(false), _loss(0.0F), _label(-1){}
    virtual ~SoftmaxLayer() {}
    virtual LayerType getType()
    {
        return LOSS_LAYER;
    }

    virtual void init(int (&params)[4]);
    virtual void setUp(const boost::shared_ptr<Data>& data);
    virtual void setLabel(int label);
    virtual void forward_cpu();
    virtual void backward_cpu();
    inline float getLoss()
    {
        return _loss;
    };

    inline bool getForecastResult()
    {
        return _forecast_success;
    }
private:
    Mode _mode;
    float _loss;
    int _label;
    bool _forecast_success;
protected:

    DISABLE_COPY_AND_ASSIGN(SoftmaxLayer);
};

}  // namespace dong

#endif  // DONG_SOFTMAX_LAYER_HPP_
