#ifndef DONG_LAYER_HPP_
#define DONG_LAYER_HPP_
#include "common.hpp"
#include "data.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>


namespace dong
{

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
    static float BASE_LEARNING_RATE;        //基准学习速率
    static LR_Policy LEARNING_RATE_POLICY;  //学习速率衰减策略
    static float GAMMA;                     //学习速率衰减常数
    static float MOMENTUM;                  //学习冲力 借助冲力逃出局部洼地
    static int CURRENT_ITER_COUNT;          //当前迭代次数
    static float POWER;
    static float WEIGHT_DECAY;              //权重衰减常数
    static float CURRENT_LEARNING_RATE;     //当前学习速率
    static int STEPSIZE;                  //每STEPSIZE次迭代，更新一次学习率
    static int BATCH_SIZE;                  //批量训练数据大小

    Layer()
    {
        _lr_mult_weight = 1.0F;
        _lr_mult_bias = 1.0F;
    }
    virtual ~Layer(){};
    virtual void init(int (&params)[4]) = 0;
    virtual void forward_cpu() = 0;
    virtual void forward();
    virtual void backward_cpu() = 0;
    virtual void backward();
    virtual LayerType getType() = 0;
    virtual boost::shared_ptr<Layer>& getTopLayer();
    virtual void setTopLayer(boost::shared_ptr<Layer>& layer);
    virtual boost::shared_ptr<Layer>& getBottomLayer();
    virtual void setBottomLayer(boost::shared_ptr<Layer>& layer);
    virtual void setLabel(int label){};
    virtual void updateWeight();
    virtual void updateBias();
    inline virtual void setName(string& name){this->_name = name;};
    inline string& getName(){return _name;};
    static float getLearningRate();

    inline virtual void setUp(const boost::shared_ptr<Data>& data)
    {
        this->_bottom_data = data;
    }

    inline virtual boost::shared_ptr<Data>& getBottomData()
    {
        return _bottom_data;
    }

    inline virtual boost::shared_ptr<Data>& getTopData()
    {
        return _top_data;
    }

    inline virtual boost::shared_ptr<Data>& getWeightData()
    {
        return _weight_data;
    }

    inline virtual boost::shared_ptr<Data>& getBiasData()
    {
        return _bias_data;
    }

    inline virtual void setLrMultWeight(float lr_mult)
    {
        this->_lr_mult_weight = lr_mult;
    }

    inline virtual void setLrMultBias(float lr_mult)
    {
        this->_lr_mult_bias = lr_mult;
    }



protected:
    virtual void forwardBase();
    virtual void backwardBase();
    static void* backwardBaseThread(void* ptr);
    static void backwardLimit(Data* _bottom_data, int offset_start, int offset_end);

    boost::shared_ptr<Data> _bottom_data;
    boost::shared_ptr<Data> _top_data;
    boost::shared_ptr<Data> _weight_data;
    boost::shared_ptr<Data> _bias_data;
    boost::shared_ptr<Layer> _top_layer;
    boost::shared_ptr<Layer> _bottom_layer;
    string _name;
    float _lr_mult_weight;
    float _lr_mult_bias;
    DISABLE_COPY_AND_ASSIGN(Layer);
};

}  // namespace dong

#endif  // DONG_LAYER_HPP_
