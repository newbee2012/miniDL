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
#include "util/gen_bmp.hpp"
using namespace std;
using namespace caffe;

namespace dong
{

void NetModelLMDB::testFromABmp(string& fileName)
{
    int channels = this->_input_shape_channels;
    int height = this->_input_shape_height;
    int width = this->_input_shape_width;
    Data batchDatas(this->_batch_size, channels, height, width, Layer::default_init_data_param);
    BYTE* pBmpBuf = BmpTool::readBmp(fileName.c_str());

    for(int h=0; h < height; ++h)
    {
        for(int w=0; w< width; ++w)
        {
            for(int c=0; c <channels; ++c)
            {
                batchDatas.get(0,c,h,w)->_value = pBmpBuf[((height - h - 1) * width + w) * 3 + channels - c - 1];
            }

        }
    }

    string path = "./test";
    batchDatas.genBmp(path);
    boost::shared_array<int> labels(new int[this->_batch_size]{0});
    this->fillDataToModel(batchDatas.get(0, 0, 0, 0), batchDatas.count(),labels);
    this->forward();

    _loss_layer->getTopData()->print();

    LossLayer* lossLayer = (LossLayer*)_loss_layer.get();
    cout<< "识别结果:"<<lossLayer->getForecastLabels()[0]<<endl;
}


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
    mydb->Open(this->_test_data_file_path, db::READ);
    db::Cursor* cursor = mydb->NewCursor();
    cursor->SeekToFirst();

    Data batchDatas(_batch_size, channels, height, width, Layer::default_init_data_param);
    boost::shared_array<int> labels(new int[_batch_size]);
    cout <<setprecision(6)<< fixed;

    int correct_sum = 0;
    int error_sum = 0;
    int size = batchDatas.offset(1, 0, 0, 0);
    for (int iter = 0; iter < _max_iter_count; ++iter)
    {
        for (int i = 0; i < _batch_size; ++i)
        {
            if(!cursor->valid())
            {
                cursor->SeekToFirst();
            }

            const string& value = cursor->value();
            Datum datum;
            datum.ParseFromString(value);

            for (int c = 0; c < channels; c++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        batchDatas.get(i, c, w, h)->_value = (BYTE)(datum.data()[c * width * height + w * height + h]);
                        labels[i] = datum.label();
                    }
                }
            }

            cursor->Next();
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
            }else
            {
                ++error_sum;
                //cout<<"index :"<<iter*_batch_size+i<<" error!"<<endl;
            }
        }

        float accuracy = (float)correct / _batch_size;
        cout << "iter:" << iter<< ", correct / count : " <<correct<<"/"<< _batch_size<< " , accuracy : "<< setprecision(6) << accuracy <<endl;
        ////////////////////////////////////////////////////////////////////////////////
    }

    int count = _batch_size * _max_iter_count;
    float accuracy = (float)correct_sum / count;
    cout<< "all iters: correct_sum, error_sum, count_sum :" <<correct_sum<<","<<error_sum<<","<< count<< ",accuracy : "<< setprecision(6) << accuracy <<endl;

    delete cursor;
    mydb->Close();
    delete mydb;

    time_t t2 = time(NULL);
    cout <<"总共耗时:"<< t2 -t1<<"秒, 预测速度:" << count /
         (t2 - t1 + 1) << " pic / s" << endl;
}

void NetModelLMDB::train()
{
    time_t t1 = time(NULL);
    int channels = _input_shape_channels;
    int height = _input_shape_height;
    int width = _input_shape_width;
    ///////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////
    db::DB* mydb = db::GetDB("lmdb");
    mydb->Open(this->_train_data_file_path, db::READ);
    db::Cursor* cursor = mydb->NewCursor();
    cursor->SeekToFirst();
    Data batchDatas(_batch_size, channels, height, width, Layer::default_init_data_param);
    boost::shared_array<int> labels(new int[_batch_size]);
    for (int iter = 0; iter < _max_iter_count; ++iter)
    {
        for (int i = 0; i < _batch_size; ++i)
        {
            if(!cursor->valid())
            {
                cout<<"Train data seek to first!"<<endl;
                cursor->SeekToFirst();
            }

            const string& value = cursor->value();
            Datum datum;
            datum.ParseFromString(value);
            for (int c = 0; c < channels; c++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        batchDatas.get(i, c, w, h)->_value = (BYTE)(datum.data()[c * width * height + w * height + h]);
                        labels[i] = datum.label();
                    }
                }
            }

            cursor->Next();
        }

        /*
        string path = "./cifar10_pics/";
        path.append(toString(iter));
        path.append("_");
        path.append(toString(labels[0]));
        batchDatas.genBmp(path);
        */
        /////////////////////////////////训练一批数据///////////////////////////////////
        this->fillDataToModel(batchDatas.get(0, 0, 0, 0), batchDatas.count(), labels);
        this->forward();
        this->backward();
        this->update();
        //////////////////////////////////////////////////////////////////////////////
        LossLayer* lossLayer = (LossLayer*)_loss_layer.get();
        cout << "iter:"<< iter << ", loss:" << setprecision(6) << fixed << lossLayer->getLoss() << ", lr_rate:" << Layer::CURRENT_LEARNING_RATE<< endl<<endl;
        if((iter+1) % 100 == 0)
        {
            this->save_model();
        }

    }

    delete cursor;
    mydb->Close();
    delete mydb;

    time_t t2 = time(NULL);
    cout <<"总共耗时:"<< t2 -t1<<"秒, 训练速度:" << (float)(_batch_size * _max_iter_count) /
         (t2 - t1 + 1) << " pic / s" << endl;
}

}
