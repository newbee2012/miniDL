#ifndef DONG_NET_MODEL_BINARY_HPP_
#define DONG_NET_MODEL_BINARY_HPP_
#include "net_model.hpp"

using namespace std;

namespace dong
{

class NetModelBinary: public NetModel
{
public:
    explicit NetModelBinary(string& modelDefineFilePath):NetModel(modelDefineFilePath) {};
    virtual ~NetModelBinary() {};
    virtual void train();
    virtual void test();
    virtual void testFromABmp(string& fileName);
    virtual void compute_mean();
};

}


#endif  // DONG_NET_MODEL_BINARY_HPP_
