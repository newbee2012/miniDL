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
using namespace std;

namespace dong
{

void NetModelMysql::train()
{
    time_t t1 = time(NULL);
    int channels = 1;
    int width = 301;
    int height = 3;
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
    Data batchDatas(_per_batch_train_count, 1, height, width, Data::CONSTANT);
    int batchLabels[_per_batch_train_count];
    cout <<setprecision(6)<< fixed;
    //训练batch_count批数据
    for (int batch = 0; batch < _batch_count; ++batch)
    {
        cout << "batch: " << batch << endl;
        //读取一批数据
        //查询数据库
        const string sql_templete =
            "select user_data, label, user_id, cur_month from train_variable_label order by user_id,cur_month limit ";
        string sql = sql_templete + toString(batch * _per_batch_train_count * height) + "," + toString(
                         _per_batch_train_count * height);
        mysql_query( &mysql, sql.c_str() );
        result = mysql_store_result( &mysql );
        MYSQL_ROW row = mysql_fetch_row( result );
        int index = 0;

        for (int n = 0; n < batchDatas.num() && NULL != row; n++)
        {
            batchLabels[n] = atoi(row[1]);

            for (int h = 0; h < batchDatas.height() && NULL != row; h++)
            {
                //cout<<"label:"<< batchLabels[n]<<",user_id:"<<row[2]<<",cur_month:"<<row[3]<< endl;
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

        //batchDatas.print();

        //训练这批数据
        for (int i = 0; i < _per_batch_train_count; i++)
        {
            int label = batchLabels[i];
            Neuron* neuron = batchDatas.get(i, 0, 0, 0);
            this->fillDataForOnceTrainForward(neuron, batchDatas.offset(1, 0, 0, 0), label);
            this->forward();
            this->backward();
            ++record_count;
            //loss_record_sum += softmaxLayer->getLoss();
            ++Layer::CURRENT_ITER_COUNT;
            Layer::CURRENT_LEARNING_RATE = Layer::getLearningRate();
        }

        float avg_loss = loss_record_sum / record_count;
        cout << "avg loss:" << setprecision(6) << fixed << avg_loss << ", lr_rate:" << Layer::CURRENT_LEARNING_RATE<<",label:"<<batchLabels[0] << endl<<endl;
        loss_record_sum = 0.0F;
        record_count = 0;
    }

    mysql_free_result( result );
    mysql_close( &mysql );
    time_t t2 = time(NULL);
    cout << "训练速度:" << _batch_count* _per_batch_train_count /
         (t2 - t1 + 1) << " pic / s" << endl;
}

}
