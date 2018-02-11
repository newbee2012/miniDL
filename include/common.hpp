#ifndef DONG_COMMON_HPP_
#define DONG_COMMON_HPP_

typedef unsigned char  BYTE;
typedef unsigned short WORD;
typedef unsigned int  DWORD;
typedef int    INT32;
// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <math.h>
#include <algorithm>
#include <float.h>
#include <boost/math/special_functions/next.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <iostream>
#include <string>
#include <assert.h>
#include "json/json.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define random(x) (rand()%x)

#define ASSERT(exp, ending_action) \
if(!(exp)) \
{ \
    cout<< "程序异常终止！ "; \
    ending_action; \
    assert(exp); \
}

using namespace std;
using namespace boost;

namespace dong
{

enum DataInitType_ {CONSTANT, RANDOM, XAVIER, GAUSSIAN, DATA_INIT_TYPE_SIZE};
static const char* DataInitNames[] = {"CONSTANT", "RANDOM", "XAVIER", "GAUSSIAN"};

enum LayerType_ {INPUT_LAYER, CONVOLUTION_LAYER, POOL_LAYER, FULL_CONNECT_LAYER, RELU_LAYER, LOSS_LAYER, LayerTypeSize};
static const char* LayerTypeNames[] = {"INPUT_LAYER", "CONVOLUTION_LAYER", "POOL_LAYER", "FULL_CONNECT_LAYER", "RELU_LAYER", "LOSS_LAYER"};

enum LR_Policy_ {FIXED, STEP, EXP, INV, MULTISTEP, POLY, SIGMOID, LR_Policy_size};
static const char* LRPolicyNames[] = {"FIXED", "STEP", "EXP", "INV", "MULTISTEP", "POLY", "SIGMOID"};

enum Mode {TRAIN, TEST, MODE_SIZE};
static const char* ModeNames[] = {"TRAIN", "TEST"};

typedef LayerType_ LayerType;
typedef LR_Policy_ LR_Policy;
typedef DataInitType_ DataInitType;



static LayerType_ STRING_TO_LAYER_TYPE(const char* name)
{
    for(int i=0; i<LayerTypeSize; ++i)
    {
        if(0==strcmp(LayerTypeNames[i],name))
        {
            return (LayerType_)i;
        }
    }

    return (LayerType_)-1;
}

static const char* LAYER_TYPE_TO_STRING(LayerType_ type)
{
    return LayerTypeNames[type];
}

static LR_Policy STRING_TO_LR_POLICY(const char* name)
{
    for(int i=0; i<LR_Policy_size; ++i)
    {
        if(0==strcmp(LRPolicyNames[i],name))
        {
            return (LR_Policy)i;
        }
    }

    return (LR_Policy)-1;
}

static Mode STRING_TO_MODE(const char* name)
{
    for(int i=0; i<MODE_SIZE; ++i)
    {
        if(0==strcmp(ModeNames[i],name))
        {
            return (Mode)i;
        }
    }

    return (Mode)-1;
}

static DataInitType STRING_TO_Data_INIT_TYPE(const char* name)
{
    for(int i=0; i<DATA_INIT_TYPE_SIZE; ++i)
    {
        if(0==strcmp(DataInitNames[i],name))
        {
            return (DataInitType)i;
        }
    }

    return (DataInitType)-1;
}


template<typename T> static string toString(const T& t)
{
    ostringstream oss;  //创建一个格式化输出流
    oss << t;           //把值传递如流中
    return oss.str();
}

}

#endif  // DONG_COMMON_HPP_
