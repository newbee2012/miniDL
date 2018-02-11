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
    inline virtual LayerType getType()
    {
        return POOL_LAYER;
    }
    virtual void init(int (&params)[6]);
    virtual void setUp(const boost::shared_ptr<Data>& data);
    virtual void forward_cpu();
    virtual void backward_cpu();

protected:
    int _kernel_h, _kernel_w;
    int _stride_h, _stride_w;
    InitType _type;

    //DISABLE_COPY_AND_ASSIGN(PoolLayer);
};

}  // namespace dong

#endif  // DONG_POOL_LAYER_HPP_
