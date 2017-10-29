#ifndef DONG_NET_MODEL_HPP_
#define DONG_NET_MODEL_HPP_
#include "common.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "layer.hpp"

using namespace std;

namespace dong
{

class NetModel
{
public:
    explicit NetModel() {};
    virtual ~NetModel() {};
    virtual void train();
    virtual void save_model(const char* filename);
    virtual void load_model(const char* filename);

private:
    boost::shared_ptr<Layer*[]> _layers;

    DISABLE_COPY_AND_ASSIGN(NetModel);
};

}


#endif  // DONG_NET_MODEL_HPP_
