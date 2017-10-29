#ifndef DONG_INPUT_LAYER_HPP_
#define DONG_INPUT_LAYER_HPP_
#include "common.hpp"
#include "layer.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>


namespace dong
{
class InputLayer: public Layer
{
public:
    explicit InputLayer() {};
    virtual ~InputLayer() {};
    inline virtual LayerType getType()
    {
        return INPUT_LAYER;
    }

    virtual void setUp(const boost::shared_ptr<Data>& data);
    virtual void forward_cpu();
    virtual void backward_cpu();

protected:

    DISABLE_COPY_AND_ASSIGN(InputLayer);
};

}  // namespace dong

#endif  // DONG_INPUT_LAYER_HPP_
