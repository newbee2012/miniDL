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
int RandomGenerator::rnd_seed;          //随机种子，-1表示使用time(0)做种子

void test2(char* p, char* q, int count1, int count2, int v)
{
    Json::Reader reader;
    Json::Value root;
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

void mysql_test()
{
    const char* host = "localhost";
    const char* user = "root";
    const char* pass = "123456";
    const char* db   = "test";
    MYSQL mysql;
    mysql_init(&mysql);
    mysql_real_connect(&mysql, host, user, pass, db, 0, NULL, 0);
    mysql_set_character_set(&mysql, "utf8");
    //查询数据库
    string sql = "select * from variable_fix limit 1;";
    mysql_query( &mysql, sql.c_str() );
    MYSQL_RES* result = NULL;
    result = mysql_store_result( &mysql );
    MYSQL_ROW row = mysql_fetch_row( result );

    while ( NULL != row )
    {
        cout << row[0] << "," << row[1] << "," << row[2] << "," << row[3] << endl;
        vector<string> rs;
        boost::split( rs, row[3], boost::is_any_of( "|" ), boost::token_compress_on );

        for (int i = 0, size = rs.size(); i < size; i++)
        {
            vector<string> group_var;
            boost::split(group_var, rs[i], boost::is_any_of( "," ), boost::token_compress_on );

            for (int i = 0, size = group_var.size(); i < size; i++)
            {
                vector<string> var;
                boost::split(var, group_var[i], boost::is_any_of( ":" ), boost::token_compress_on );
                double v = atof(var[1].c_str());
                cout << v << endl;
            }
        }

        row = mysql_fetch_row( result );
    }

    mysql_free_result( result );
    mysql_close( &mysql );
    return;
}

void train2(int argc, char* argv[])
{
    /*
    int batch_count = 1;
    int per_batch_iter_count = 1;
    int per_iter_train_count = 1;
    Layer::BASE_LEARNING_RATE = 0.0001F;
    Layer::LEARNING_RATE_POLICY = INV;
    Layer::GAMMA = 0.0001F;
    Layer::MOMENTUM = 0.9F;
    Layer::POWER = 0.75F;
    Layer::WEIGHT_DECAY = 0.0005F;
    Layer::CURRENT_ITER_COUNT = 0;
    Layer::STEPSIZE = 100;
    RandomGenerator::rnd_seed = -1;

    if (argc >= 5) {
        batch_count = atoi(argv[1]);
        per_batch_iter_count = atoi(argv[2]);
        per_iter_train_count = atoi(argv[3]);
        Layer::BASE_LEARNING_RATE = atof(argv[4]);
        Layer::CURRENT_LEARNING_RATE = Layer::getLearningRate();
    }

    if (argc == 6) {
        RandomGenerator::rnd_seed = atoi(argv[5]);
    }

    if (RandomGenerator::rnd_seed == -1) {
        RandomGenerator::rnd_seed = (int)time(0);
    }

    time_t t1 = time(NULL);
    srand(RandomGenerator::rnd_seed);
    int channels = 1;
    int width = 301;
    int height = 3;
    ///////////////////////////////////////////////////////////////////////////
    boost::shared_array<Neuron> inputImage(new Neuron[height * width]);
    boost::shared_ptr<Data> inputData(new Data(1, channels, height, width));
    inputData->setUp(inputImage);
    //L1.inputLayer
    boost::shared_ptr<InputLayer> inputLayer(new InputLayer());
    inputLayer->setUp(inputData);
    //L1.ConvLayer
    boost::shared_ptr<ConvLayer> convLayer(new ConvLayer());
    //convLayer->init(int[3]{100, 1, width});
    convLayer->setUp(inputLayer->getTopData());
    //L2.FullConnectLayer
    boost::shared_ptr<FullConnectLayer> fullConnectLayer(new FullConnectLayer());
    //fullConnectLayer->init(100);
    fullConnectLayer->setUp(convLayer->getTopData());
    //L3.reluLayer
    boost::shared_ptr<ReluLayer> reluLayer(new ReluLayer());
    //reluLayer->init();
    reluLayer->setUp(fullConnectLayer->getTopData());
    //L4.FullConnectLayer
    boost::shared_ptr<FullConnectLayer> fullConnectLayer2(new FullConnectLayer());
    //fullConnectLayer2->init(20);
    fullConnectLayer2->setUp(reluLayer->getTopData());
    //L5.FullConnectLayer
    boost::shared_ptr<ReluLayer> reluLayer2(new ReluLayer());
    //reluLayer2->init();
    reluLayer2->setUp(fullConnectLayer2->getTopData());
    boost::shared_ptr<FullConnectLayer> fullConnectLayer3(new FullConnectLayer());
    //fullConnectLayer3->init(10);
    fullConnectLayer3->setUp(reluLayer->getTopData());
    boost::shared_ptr<ReluLayer> reluLayer3(new ReluLayer());
    //reluLayer3->init();
    reluLayer3->setUp(fullConnectLayer3->getTopData());
    boost::shared_ptr<FullConnectLayer> fullConnectLayer4(new FullConnectLayer());
    //fullConnectLayer4->init(2);
    fullConnectLayer4->setUp(reluLayer->getTopData());
    //L9.SoftmaxLayer
    boost::shared_ptr<SoftmaxLayer> softmaxLayer(new SoftmaxLayer(dong::TRAIN));
    //softmaxLayer->init();
    softmaxLayer->setUp(fullConnectLayer3->getTopData());
    float loss_record_sum = 0.0F;
    int record_count = 0;
    Data batchDatas(per_iter_train_count, 1, height, width, Data::CONSTANT);
    int batchLabels[per_iter_train_count];
    /////////////////////////////////////////////////////
    const char* host = "localhost";
    const char* user = "root";
    const char* pass = "123456";
    const char* db   = "test";
    MYSQL mysql;
    mysql_init(&mysql);
    mysql_real_connect(&mysql, host, user, pass, db, 0, NULL, 0);
    mysql_set_character_set(&mysql, "utf8");
    MYSQL_RES* result = NULL;

    //训练batch_count批数据
    for (int batch = 0; batch < batch_count; ++batch) {
        cout << "batch: " << batch << endl;
        //读取一批数据
        //查询数据库
        const string sql_templete =
            "select user_data, label, user_id, cur_month from train_variable_label order by user_id,cur_month limit ";
        string sql = sql_templete + toString(batch * per_iter_train_count * height) + "," + toString(
                         per_iter_train_count * height);
        mysql_query( &mysql, sql.c_str() );
        result = mysql_store_result( &mysql );
        MYSQL_ROW row = mysql_fetch_row( result );
        int index = 0;

        for (int n = 0; n < batchDatas.num() && NULL != row; n++) {
            batchLabels[n] = atoi(row[1]);

            for (int h = 0; h < batchDatas.height() && NULL != row; h++) {
                //cout<<"label:"<< batchLabels[n]<<",user_id:"<<row[2]<<",cur_month:"<<row[3]<< endl;
                //cout<<"value:";
                vector<string> rs;
                boost::split( rs, row[0], boost::is_any_of( "|" ), boost::token_compress_on );

                for (int j = 0; j < rs.size(); j++) {
                    vector<string> group_var;
                    boost::split(group_var, rs[j], boost::is_any_of( "," ), boost::token_compress_on );

                    for (int k = 0; k < group_var.size(); k++) {
                        vector<string> var;
                        boost::split(var, group_var[k], boost::is_any_of( ":" ), boost::token_compress_on );
                        float v = atof(var[1].c_str());
                        batchDatas.get(n, 0, h, k)->_value = v;
                        //cout<<v<<",";
                    }
                }

                //cout<<endl;
                row = mysql_fetch_row( result );
            }

            //cout<<"---------------------------------------"<<endl;
        }

        //每一批数据迭代per_batch_iter_count次
        for (int iter = 0; iter < per_batch_iter_count; ++iter) {
            //训练这批数据
            for (int i = 0; i < per_iter_train_count; i++) {
                int label = batchLabels[i];
                Neuron* neuron = batchDatas.get(i, 0, 0, 0);
                Neuron* inputNeuron = inputImage.get();

                for (int j = 0; j < height * width; ++j) {
                    inputNeuron[j]._value = neuron[j]._value;
                }

                convLayer->forward();
                fullConnectLayer->forward();
                reluLayer->forward();
                fullConnectLayer2->forward();
                reluLayer2->forward();
                fullConnectLayer3->forward();
                reluLayer3->forward();
                fullConnectLayer4->forward();
                softmaxLayer->setLabel(label);
                softmaxLayer->forward();
                softmaxLayer->backward();
                fullConnectLayer4->backward();
                reluLayer3->backward();
                fullConnectLayer3->backward();
                reluLayer2->backward();
                fullConnectLayer2->backward();
                reluLayer->backward();
                fullConnectLayer->backward();
                convLayer->backward();
                ++record_count;
                loss_record_sum += softmaxLayer->getLoss();
                ++Layer::CURRENT_ITER_COUNT;
                Layer::CURRENT_LEARNING_RATE = Layer::getLearningRate();
            }

            float avg_loss = loss_record_sum / record_count;
            cout << "avg loss:" << setprecision(8) << fixed << avg_loss << ", lr_rate:" << Layer::CURRENT_LEARNING_RATE<<",label:"<<batchLabels[0] << endl;
            loss_record_sum = 0.0F;
            record_count = 0;
        }
    }

    mysql_free_result( result );
    mysql_close( &mysql );
    time_t t2 = time(NULL);
    cout << "训练速度:" << batch_count* per_batch_iter_count* per_iter_train_count /
         (t2 - t1 + 1) << " pic / s" << endl;
         */
}

int main(int argc, char* argv[])
{
    Neuron* c = new Neuron[5];
    c[0]._value = 1;
    c[1]._value = 2;
    c[2]._value = 3;
    c[3]._value = 4;
    c[4]._value = 5;

    cout<<(*c++)._value<<endl;
    cout<<(*c++)._value<<endl;
    int batch_count = 1;
    int per_iter_train_count = 1;
    Layer::BASE_LEARNING_RATE = 0.0001F;
    Layer::LEARNING_RATE_POLICY = INV;
    Layer::GAMMA = 0.0001F;
    Layer::MOMENTUM = 0.9F;
    Layer::POWER = 0.75F;
    Layer::WEIGHT_DECAY = 0.0005F;
    Layer::CURRENT_ITER_COUNT = 0;
    Layer::STEPSIZE = 100;
    RandomGenerator::rnd_seed = -1;

    if (argc >= 4) {
        batch_count = atoi(argv[1]);
        per_iter_train_count = atoi(argv[2]);
        Layer::BASE_LEARNING_RATE = atof(argv[3]);
        Layer::CURRENT_LEARNING_RATE = Layer::getLearningRate();
    }

    if (argc == 5) {
        RandomGenerator::rnd_seed = atoi(argv[4]);
    }

    if (RandomGenerator::rnd_seed == -1) {
        RandomGenerator::rnd_seed = (int)time(0);
    }
    srand(RandomGenerator::rnd_seed);

    NetModel* netMode = new NetModelMysql();
    netMode->load_model("/home/chendejia/workspace/github/miniDL/net_model.json");
    netMode->train();
    delete netMode;
    cout << "Hello world!" << endl;
    return 0;
}
