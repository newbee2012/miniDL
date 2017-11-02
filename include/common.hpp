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

#include <math.h>
#include <algorithm>
#include <float.h>
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <assert.h>
#include "json/json.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define random(x) (rand()%x)

#define ASSERT(exp, ending_action) \
if(!(exp)) \
{ \
    cout<< "程序异常终止！" << "原因:"; \
    ending_action; \
    assert(exp); \
}



using namespace std;
using namespace boost;

namespace dong
{
const int LayerTypeSize = 6;
enum LayerType_ {INPUT_LAYER, CONVOLUTION_LAYER, POOL_LAYER, FULL_CONNECT_LAYER, RELU_LAYER, SOFTMAX_LAYER};
static const char* LayerTypeNames[] = {"INPUT_LAYER", "CONVOLUTION_LAYER", "POOL_LAYER", "FULL_CONNECT_LAYER", "RELU_LAYER", "SOFTMAX_LAYER"};

enum LR_Policy_ {FIXED, STEP, EXP, INV, MULTISTEP, POLY, SIGMOID};
typedef LayerType_ LayerType;
typedef LR_Policy_ LR_Policy;
enum Mode {TRAIN, TEST};

static LayerType_ STRING_TO_LAYER_TYPE(const char* name)
{
    for(int i=0;i<LayerTypeSize;++i)
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

}

#endif  // DONG_COMMON_HPP_
