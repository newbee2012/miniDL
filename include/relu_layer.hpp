#ifndef DONG_RELU_LAYER_HPP_
#define DONG_RELU_LAYER_HPP_
#include "common.hpp"
#include "layer.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>


namespace dong
{
class ReluLayer: public Layer
{
public:
    explicit ReluLayer() {};
    virtual ~ReluLayer() {};
    inline virtual LayerType getType()
    {
        return RELU_LAYER;
    }
    virtual void init(int (&params)[6]);
    virtual void setUp(const boost::shared_ptr<Data>& data);
    virtual void forward_cpu();
    virtual void backward_cpu();

protected:

    DISABLE_COPY_AND_ASSIGN(ReluLayer);
};

}  // namespace dong

#endif  // DONG_RELU_LAYER_HPP_
