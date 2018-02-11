#include "common.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "util/math_utils.hpp"
#include <mysql/mysql.h>
#include "net_model_mysql.hpp"
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include "softmax_layer.hpp"
using namespace std;

namespace dong
{

void NetModelMysql::train()
{
    time_t t1 = time(NULL);
    int channels = _input_shape_channels;
    int height = _input_shape_height;
    int width = _input_shape_width;
    ///////////////////////////////////////////////////////////////////////////
    float loss_record_sum = 0.0F;
    int record_count = 0;

    /////////////////////////////////////////////////////
    const char* host = "127.0.0.1";
    const char* user = "root";
    const char* pass = "123456";
    const char* db   = "test";
    MYSQL mysql;
    mysql_init(&mysql);
    mysql_real_connect(&mysql, host, user, pass, db, 0, NULL, 0);
    mysql_set_character_set(&mysql, "utf8");
    MYSQL_RES* result = NULL;
    Data batchDatas(_batch_size, 1, height, width, CONSTANT);
    boost::shared_array<int> labels(new int[_batch_size]);
    cout <<setprecision(6)<< fixed;
    //训练batch_count批数据
    int correct_sum = 0;
    for (int iter = 0; iter < _max_iter_count; ++iter)
    {
        //读取一批数据
        //查询数据库
        const string sql_templete =
            "select user_data, label, user_id, cur_month from train_variable_label order by user_id,cur_month limit ";
        string sql = sql_templete + toString(iter * _batch_size * height) + "," + toString(
                         _batch_size * height);
        mysql_query( &mysql, sql.c_str() );
        result = mysql_store_result( &mysql );
        MYSQL_ROW row = mysql_fetch_row( result );
        int index = 0;

        for (int n = 0; n < batchDatas.num() && NULL != row; n++)
        {
            labels[n] = atoi(row[1]);

            for (int h = 0; h < batchDatas.height() && NULL != row; h++)
            {
                //cout<<"label:"<< labels[n]<<",user_id:"<<row[2]<<",cur_month:"<<row[3]<< endl;
                vector<string> rs;
                boost::split( rs, row[0], boost::is_any_of( "|" ), boost::token_compress_on );
                int count = 0;
                for (int j = 0; j < rs.size(); j++)
                {
                    vector<string> group_var;
                    boost::split(group_var, rs[j], boost::is_any_of( "," ), boost::token_compress_on );
                    for (int k = 0; k < group_var.size(); k++)
                    {
                        vector<string> var;
                        boost::split(var, group_var[k], boost::is_any_of( ":" ), boost::token_compress_on );
                        float v = atof(var[1].c_str());
                        batchDatas.get(n, 0, h, count)->_value = v;
                        count++;
                    }
                }

                row = mysql_fetch_row( result );
            }
        }

        /////////////////////////////////训练一批数据///////////////////////////////////
        this->fillDataToModel(batchDatas.get(0, 0, 0, 0), batchDatas.count(), labels);
        this->forward();
        this->backward();
        this->update();
        //////////////////////////////////////////////////////////////////////////////
        LossLayer* lossLayer = (LossLayer*)_loss_layer.get();
        cout << "iter:"<< iter << ", loss:" << setprecision(6) << fixed << lossLayer->getLoss() << ", lr_rate:" << Layer::CURRENT_LEARNING_RATE<< endl<<endl;
        if(iter % 100 == 0){
            this->save_model();
        }
    }

    mysql_free_result( result );
    mysql_close( &mysql );
    time_t t2 = time(NULL);
    cout <<"总共耗时:"<< t2 -t1<<"秒, 训练速度:" << (float)(_batch_size * _max_iter_count) /
         (t2 - t1 + 1) << " pic / s" << endl;
}

void NetModelMysql::test()
{
    time_t t1 = time(NULL);
    int channels = _input_shape_channels;
    int height = _input_shape_height;
    int width = _input_shape_width;
    ///////////////////////////////////////////////////////////////////////////
    float loss_record_sum = 0.0F;
    int record_count = 0;

    /////////////////////////////////////////////////////
    const char* host = "127.0.0.1";
    const char* user = "root";
    const char* pass = "123456";
    const char* db   = "test";
    MYSQL mysql;
    mysql_init(&mysql);
    mysql_real_connect(&mysql, host, user, pass, db, 0, NULL, 0);
    mysql_set_character_set(&mysql, "utf8");
    MYSQL_RES* result = NULL;
    Data batchDatas(_batch_size, 1, height, width, CONSTANT);
    boost::shared_array<int> labels(new int[_batch_size]);
    cout <<setprecision(6)<< fixed;
    //训练batch_count批数据
    int correct_sum = 0;
    for (int iter = 0; iter < _max_iter_count; ++iter)
    {
        //读取一批数据
        //查询数据库
        const string sql_templete =
            "select user_data, label, user_id, cur_month from train_variable_label order by user_id,cur_month limit ";
        string sql = sql_templete + toString(iter * _batch_size * height) + "," + toString(
                         _batch_size * height);
        mysql_query( &mysql, sql.c_str() );
        result = mysql_store_result( &mysql );
        MYSQL_ROW row = mysql_fetch_row( result );
        int index = 0;

        for (int n = 0; n < batchDatas.num() && NULL != row; n++)
        {
            labels[n] = atoi(row[1]);

            for (int h = 0; h < batchDatas.height() && NULL != row; h++)
            {
                //cout<<"label:"<< labels[n]<<",user_id:"<<row[2]<<",cur_month:"<<row[3]<< endl;
                vector<string> rs;
                boost::split( rs, row[0], boost::is_any_of( "|" ), boost::token_compress_on );
                int count = 0;
                for (int j = 0; j < rs.size(); j++)
                {
                    vector<string> group_var;
                    boost::split(group_var, rs[j], boost::is_any_of( "," ), boost::token_compress_on );
                    for (int k = 0; k < group_var.size(); k++)
                    {
                        vector<string> var;
                        boost::split(var, group_var[k], boost::is_any_of( ":" ), boost::token_compress_on );
                        float v = atof(var[1].c_str());
                        batchDatas.get(n, 0, h, count)->_value = v;
                        count++;
                    }
                }

                row = mysql_fetch_row( result );
            }
        }

        /////////////////////////////测试这批数据，统计准确率/////////////////////////////////////////
        LossLayer* lossLayer = (LossLayer*)_loss_layer.get();
        this->fillDataToModel(batchDatas.get(0, 0, 0, 0), batchDatas.count(), labels);
        this->forward();

        int correct = 0;
        boost::shared_array<int>& forecastLabels = lossLayer->getForecastLabels();
        for(int i=0; i < _batch_size; ++i)
        {
            if(labels[i] == forecastLabels[i])
            {
                ++correct;
                ++correct_sum;
            }
        }

        float accuracy = (float)correct / _batch_size;
        cout << "iter:" << iter<< ", correct / count : " <<correct<<"/"<< _batch_size<< " , accuracy : "<< setprecision(
                 6) << accuracy <<endl;
    }

    int count = _batch_size * _max_iter_count;
    float accuracy = (float)correct_sum / count;
    cout<< "all iters: correct_sum / count_sum:" <<correct_sum<<"/"<< count<< " , accuracy : "<< setprecision(
            6) << accuracy <<endl;

    mysql_free_result( result );
    mysql_close( &mysql );
    time_t t2 = time(NULL);
    cout <<"总共耗时:"<< t2 -t1<<"秒, 预测速度:" << count /
         (t2 - t1 + 1) << " pic / s" << endl;
}
}
