#ifndef DONG_POOL_LAYER_HPP_
#define DONG_POOL_LAYER_HPP_
#include "common.hpp"
#include "layer.hpp"
#include<memory>
#include <boost/shared_ptr.hpp>
#include <vector>


namespace dong
{
class PoolLayer: public Layer
{
public:
    enum InitType {MAX,AVE};
    PoolLayer() {};
    virtual ~PoolLayer() {};
    virtual LayerType getType()=0;
    virtual void init(int (&params)[6]);
    virtual void setUp(const boost::shared_ptr<Data>& data);
    virtual void forward_cpu()=0;
    virtual void backward_cpu();

protected:
    int _kernel_h, _kernel_w;
    int _stride_h, _stride_w;
    InitType _type;

    DISABLE_COPY_AND_ASSIGN(PoolLayer);
};

class MaxPoolLayer: public PoolLayer
{
public:
    virtual LayerType getType();
    virtual void forward_cpu();
};

class AvePoolLayer: public PoolLayer
{
public:
    virtual LayerType getType();
    virtual void forward_cpu();
};

}  // namespace dong

#endif  // DONG_POOL_LAYER_HPP_
