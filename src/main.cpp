#include <iostream>
#include "util/gen_bmp.hpp"
#include "input_layer.hpp"
#include "conv_layer.hpp"
#include "pool_layer.hpp"
#include "full_connect_layer.hpp"
#include "relu_layer.hpp"
#include "softmax_layer.hpp"
#include <time.h>
#include <boost/shared_ptr.hpp>
#include <pthread.h>
#include "util/math_utils.hpp"
#include <mysql/mysql.h>
#include <string>
#include <vector>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include "net_model_mysql.hpp"
#include "net_model_lmdb.hpp"
#include "net_model.hpp"

using namespace std;
using namespace dong;
int sum = 0;


float Layer::BASE_LEARNING_RATE;        //基准学习数率
dong::LR_Policy Layer::LEARNING_RATE_POLICY;  //学习数率衰减策略
float Layer::GAMMA;                     //学习速率衰减常数
float Layer::MOMENTUM;                  //学习冲力 借助冲力逃出局部洼地
int Layer::CURRENT_ITER_COUNT;                  //当前迭代次数
float Layer::POWER;
float Layer::WEIGHT_DECAY;              //权重衰减常数
float Layer::CURRENT_LEARNING_RATE;
int Layer::STEPSIZE;
int Layer::BATCH_SIZE;

int RandomGenerator::rnd_seed;          //随机种子，-1表示使用time(0)做种子

void test2(char* p, char* q, int count1, int count2, int v)
{
    if (count1 + count2 == 0)
    {
        cout << p << endl;
        sum++;
        return;
    }

    if (v >= 0 && count1 > 0)
    {
        *q = '(';
        test2(p, q + 1, count1 - 1, count2, v + 1);
    }

    if (count2 > 0)
    {
        *q = ')';
        test2(p, q + 1, count1, count2 - 1, v - 1);
    }
}

int64_t mypow(int64_t n, int exp)
{
    int64_t value = 1;
    int i;
    for(i=1;i<=exp; ++i)
    {
        value *= n;
    }

    return value;
}

int64_t count3(int len)
{
    int64_t t=1;
    int i;
    for(i=2; i<=len; ++i)
    {
        t=t*10+mypow(10,i-1)/2;
    }

    return t;
}

int64_t test(char* x)
{
    char c = x[0];

    int len = strlen(x);
    int num = c-'0';

    if(len ==1)
    {
        if(num >= 3)
        {
            return 1;
        }else
        {
            return 0;
        }
    }

    int64_t t= count3(len-1);
    cout<<"count3:"<<t<<",len:"<<len-1<<endl;
    x++;
    if(num > 3)
    {
        return num * t + test(x)+ mypow(10,len-1)/2;
    }
    else if(num == 3)
    {
        return num * t + (atoi(x)+1)/2 + test(x);
    }
    else
    {
        return num * t + test(x);
    }
}


void solution(char *line)
{
    int a;
    // 在此处理单行测试数据
    sscanf(line,"%d",&a);
    // 打印处理结果
    printf("%lld\n", test(line));
}


int main(int argc, char* argv[])
{
    RandomGenerator::rnd_seed = -1;
    if (argc == 2) {
        RandomGenerator::rnd_seed = atoi(argv[1]);
    }

    if (RandomGenerator::rnd_seed == -1) {
        RandomGenerator::rnd_seed = (int)time(0);
    }

    srand(RandomGenerator::rnd_seed);

    string modelDefileName = "/home/chendejia/workspace/github/miniDL/net_model_define_mnist.json";
    cout<<"load model file: "<<modelDefileName<<endl;
    NetModelLMDB* netMode = new NetModelLMDB(modelDefileName);
    //netMode->load_model();
    //string fileName = "/home/chendejia/workspace/github/miniDL/bin/Release/7.bmp";
    //netMode->testFromABmp(fileName);
    netMode->run();
    delete netMode;

    //cout << "Hello world!" << endl;
    //solution(argv[1]);
    return 0;
}
