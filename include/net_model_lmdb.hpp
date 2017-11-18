#ifndef DONG_NET_MODEL_LMDB_HPP_
#define DONG_NET_MODEL_LMDB_HPP_
#include "net_model.hpp"

using namespace std;

namespace dong
{

class NetModelLMDB: public NetModel
{
public:
    explicit NetModelLMDB(string& modelDefineFilePath):NetModel(modelDefineFilePath) {};
    virtual ~NetModelLMDB() {};
    virtual void train();
    virtual void test();
    virtual void testFromABmp(string& fileName);
};

}


#endif  // DONG_NET_MODEL_LMDB_HPP_
