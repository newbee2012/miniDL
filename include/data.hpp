#ifndef DONG_DATA_LAYER_HPP_
#define DONG_DATA_LAYER_HPP_
#include "common.hpp"
#include "neuron.hpp"
#include <vector>


namespace dong
{

class InitDataParam
{
public:
    InitDataParam():_initType(dong::CONSTANT),_constant_value(0.0F),_gaussian_std(1.0F),_gaussian_mean(0.0F){}

    DataInitType _initType;
    float _constant_value;
    float _gaussian_std;
    float _gaussian_mean;
};

class Data
{
public:
    explicit Data(int num, int channels, int height, int width, boost::shared_ptr<InitDataParam>& param);
    explicit Data(int num, int channels, int height, int width);
    void setUp(const boost::shared_array<Neuron>& neurons);
    void print();
    void printDiff();
    void genBmp(string& filePathBase);
    void clearDiff();
    void clearValue();

    inline int count()
    {
        return _num * _channels * _width * _height;
    }

    inline Neuron* get(int offset)
    {
        return &_neurons[offset];
    };

    inline Neuron* get(int n, int c, int h, int w)
    {
        return get(offset(n, c, h, w));
    };

    inline int offset(int n, int c, int h, int w) const
    {
        return ((n * channels() + c) * height() + h) * width() + w;
    }

    inline int num() const
    {
        return _num;
    }

    inline int channels() const
    {
        return _channels;
    }

    inline int height() const
    {
        return _height;
    }

    inline int width() const
    {
        return _width;
    }

protected:
    boost::shared_array<Neuron> _neurons;
    int _num;
    int _channels;
    int _height;
    int _width;

    DISABLE_COPY_AND_ASSIGN(Data);
};

}  // namespace dong

#endif  // DONG_DATA_LAYER_HPP_
