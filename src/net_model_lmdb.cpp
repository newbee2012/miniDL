#include "common.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "util/db.hpp"
#include "util/math_utils.hpp"
#include "net_model_lmdb.hpp"
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include "softmax_layer.hpp"

using namespace std;
using namespace caffe;

namespace dong
{
void NetModelLMDB::test()
{
    time_t t1 = time(NULL);
    int channels = _input_shape_channels;
    int height = _input_shape_height;
    int width = _input_shape_width;
    ///////////////////////////////////////////////////////////////////////////
    float loss_record_sum = 0.0F;
    int record_count = 0;

    /////////////////////////////////////////////////////
    db::DB* mydb = db::GetDB("lmdb");
    mydb->Open("/home/chendejia/workspace/github/dong/data/mnist_test_lmdb", db::READ);
    db::Cursor* cursor = mydb->NewCursor();
    cursor->SeekToFirst();

    Data batchDatas(_per_batch_train_count, 1, height, width, Data::CONSTANT);
    int batchLabels[_per_batch_train_count];
    cout <<setprecision(6)<< fixed;

    int success_count = 0;
    int test_count = 0;
    //训练batch_count批数据
    for (int batch = 0; batch < _batch_count; ++batch)
    {
        //读取一批数据
        for (int i = 0; i < _per_batch_train_count && cursor->valid(); i++, cursor->Next()) {
            const string& value = cursor->value();
            Datum datum;
            datum.ParseFromString(value);

            for (int c = 0; c < channels; c++) {
                for (int w = 0; w < width; w++) {
                    for (int h = 0; h < height; h++) {
                        batchDatas.get(i, c, w, h)->_value = (BYTE)(datum.data()[w * height + h]);
                        batchLabels[i] = datum.label();
                    }
                }
            }
        }

        //训练这批数据
        for (int i = 0; i < _per_batch_train_count; i++)
        {
            int label = batchLabels[i];
            Neuron* neuron = batchDatas.get(i, 0, 0, 0);
            this->fillDataForOnceTrainForward(neuron, batchDatas.offset(1, 0, 0, 0), label);
            this->forward();
            //this->backward();
            ++test_count;
            LossLayer* lossLayer = (LossLayer*)_loss_layer.get();
            if(lossLayer->getForecastResult())
            {
                ++success_count;
            }

            //cout<<"label:"<<label<<",forecastResult:"<< lossLayer->getForecastResult()<<endl;
            //_loss_layer->getTopData()->print();
            //cout<<endl;

            loss_record_sum += lossLayer->getLoss();
            ++record_count;

            ++Layer::CURRENT_ITER_COUNT;
            Layer::CURRENT_LEARNING_RATE = Layer::getLearningRate();
        }

        float success_rate = (float)success_count / (float)test_count;
        cout << "success_count / test_count : " <<success_count<<"/"<< test_count<< ",success_rate:"<< setprecision(6) << success_rate <<endl;

        loss_record_sum = 0.0F;
        record_count = 0;
    }

    float success_rate = (float)success_count / (float)test_count;
    cout << "success_count / test_count : " <<success_count<<"/"<< test_count<< ",success_rate:"<< setprecision(6) << success_rate <<endl;

    delete cursor;
    mydb->Close();
    delete mydb;

    time_t t2 = time(NULL);
    cout <<"总共耗时:"<< t2 -t1<<"秒, 预测速度:" << test_count /
         (t2 - t1 + 1) << " pic / s" << endl;
}

void NetModelLMDB::train()
{
    time_t t1 = time(NULL);
    int channels = _input_shape_channels;
    int height = _input_shape_height;
    int width = _input_shape_width;
    ///////////////////////////////////////////////////////////////////////////
    float loss_record_sum = 0.0F;
    int record_count = 0;

    /////////////////////////////////////////////////////
    db::DB* mydb = db::GetDB("lmdb");
    mydb->Open("/home/chendejia/workspace/github/dong/data/mnist_train_lmdb", db::READ);
    db::Cursor* cursor = mydb->NewCursor();
    cursor->SeekToFirst();

    Data batchDatas(_per_batch_train_count, 1, height, width, Data::CONSTANT);
    int batchLabels[_per_batch_train_count];
    cout <<setprecision(6)<< fixed;

    int train_count = 0;
    //训练batch_count批数据
    for (int batch = 0; batch < _batch_count; ++batch)
    {
        //读取一批数据
        for (int i = 0; i < _per_batch_train_count && cursor->valid(); i++, cursor->Next()) {
            const string& value = cursor->value();
            Datum datum;
            datum.ParseFromString(value);

            for (int c = 0; c < channels; c++) {
                for (int w = 0; w < width; w++) {
                    for (int h = 0; h < height; h++) {
                        batchDatas.get(i, c, w, h)->_value = (BYTE)(datum.data()[w * height + h]);
                        batchLabels[i] = datum.label();
                    }
                }
            }
        }

        //训练这批数据
        for (int i = 0; i < _per_batch_train_count; i++)
        {
            int label = batchLabels[i];
            Neuron* neuron = batchDatas.get(i, 0, 0, 0);
            this->fillDataForOnceTrainForward(neuron, batchDatas.offset(1, 0, 0, 0), label);
            this->forward();
            this->backward();
            ++record_count;
            ++train_count;
            LossLayer* lossLayer = (LossLayer*)_loss_layer.get();
            loss_record_sum += lossLayer->getLoss();
            ++Layer::CURRENT_ITER_COUNT;
            Layer::CURRENT_LEARNING_RATE = Layer::getLearningRate();
        }

        float avg_loss = loss_record_sum / record_count;
        //this->save_model();
        cout << "avg loss:" << setprecision(6) << fixed << avg_loss << ", lr_rate:" << Layer::CURRENT_LEARNING_RATE<<",label:"<<batchLabels[0] << endl<<endl;

        loss_record_sum = 0.0F;
        record_count = 0;
    }

    delete cursor;
    mydb->Close();
    delete mydb;

    time_t t2 = time(NULL);
    cout <<"总共耗时:"<< t2 -t1<<"秒, 训练速度:" << train_count /
         (t2 - t1 + 1) << " pic / s" << endl;
}

}
