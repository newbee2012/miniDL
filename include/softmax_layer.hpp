#ifndef DONG_SOFTMAX_LAYER_HPP_
#define DONG_SOFTMAX_LAYER_HPP_
#include "common.hpp"
#include "layer.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>


namespace dong
{
class SoftmaxLayer: public Layer
{
public:

    explicit SoftmaxLayer(Mode mode): _mode(mode), _loss(0.0F), _label(-1) {}
    virtual ~SoftmaxLayer() {}
    virtual LayerType getType()
    {
        return SOFTMAX_LAYER;
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
private:
    Mode _mode;
    float _loss;
    int _label;
protected:

    DISABLE_COPY_AND_ASSIGN(SoftmaxLayer);
};

}  // namespace dong

#endif  // DONG_SOFTMAX_LAYER_HPP_
