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
    virtual boost::shared_array<int>&  getForecastLabels() = 0;
};

class SoftmaxLayer: public LossLayer
{
public:

    explicit SoftmaxLayer(Mode mode):_mode(mode), _loss(0.0F){}
    virtual ~SoftmaxLayer() {}
    virtual LayerType getType()
    {
        return LOSS_LAYER;
    }

    virtual void init(int (&params)[6]);
    virtual void setUp(const boost::shared_ptr<Data>& data);
    virtual void setLabels(boost::shared_array<int>& labels);
    virtual void forward_cpu();
    virtual void backward_cpu();
    inline float getLoss()
    {
        return _loss;
    };

    inline boost::shared_array<int>& getForecastLabels()
    {
        return _forecast_labels;
    }
private:
    Mode _mode;
    float _loss;
    boost::shared_array<int> _labels;
    boost::shared_array<int> _forecast_labels;
    int _num;
protected:

    DISABLE_COPY_AND_ASSIGN(SoftmaxLayer);
};

}  // namespace dong

#endif  // DONG_SOFTMAX_LAYER_HPP_
