#include <iostream>
#include "util/math_utils.hpp"
#include "net_model_mysql.hpp"
#include "net_model_lmdb.hpp"
#include "net_model.hpp"

using namespace std;
using namespace dong;

float Layer::BASE_LEARNING_RATE;        //基准学习数率
dong::LR_Policy Layer::LEARNING_RATE_POLICY;  //学习数率衰减策略
float Layer::GAMMA;                     //学习速率衰减常数
float Layer::MOMENTUM;                  //学习冲力 借助冲力逃出局部洼地
int Layer::CURRENT_ITER_COUNT;                  //当前迭代次数
float Layer::POWER;
float Layer::WEIGHT_DECAY;              //权重衰减常数
float Layer::CURRENT_LEARNING_RATE;
int Layer::STEPSIZE;
int Layer::FORWARD_THREAD_COUNT;
int Layer::BACKWARD_THREAD_COUNT;
boost::shared_ptr<rng_t> RandomGenerator::engine;
void runModel(string modelFilePath)
{
    NetModelLMDB* netMode = new NetModelLMDB(modelFilePath);
    netMode->load_model();
    netMode->run();
    delete netMode;
}

void forecastBmp(string modelFilePath, string picFilePath)
{
    NetModelLMDB* netMode = new NetModelLMDB(modelFilePath);
    netMode->load_model();
    netMode->testFromABmp(picFilePath);
    delete netMode;
}


int main(int argc, char* argv[])
{
    int rnd_seed = (int)time(0);
    if (argc >= 2)
    {
        string modelDefileName = argv[1];
        if (argc >= 3)
        {
            string model = argv[2];
            if(model == "-i")
            {
                string bmpFilePath = argv[3];
                forecastBmp(modelDefileName, bmpFilePath);
                return 0;
            }
            else
            {
                rnd_seed = atoi(argv[2]);
            }
        }

        RandomGenerator::init_engine(rnd_seed);
        //srand(RandomGenerator::rnd_seed);
        runModel(modelDefileName);
    }
    else
    {
        cout<<"Error! Need parameters!"<<endl;
    }


    return 0;
}
