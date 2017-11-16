#ifndef DONG_NET_MODEL_MYSQL_HPP_
#define DONG_NET_MODEL_MYSQL_HPP_
#include "net_model.hpp"

using namespace std;

namespace dong
{

class NetModelMysql: public NetModel
{
public:
    explicit NetModelMysql(string& modelDefineFilePath):NetModel(modelDefineFilePath) {};
    virtual ~NetModelMysql() {};
    virtual void train();
};

}


#endif  // DONG_NET_MODEL_MYSQL_HPP_
