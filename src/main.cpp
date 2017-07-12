#include <iostream>
#include "util/gen_bmp.hpp"
#include "input_layer.hpp"
#include "conv_layer.hpp"
#include "pool_layer.hpp"
#include "full_connect_layer.hpp"
#include "relu_layer.hpp"
#include "softmax_layer.hpp"
#include "util/db.hpp"
#include <time.h>
#include <boost/shared_ptr.hpp>
#include <pthread.h>
using namespace std;
using namespace caffe;
using namespace dong;

float Layer::BASE_LEARNING_RATE;

int sum = 0;

void test2(char* p, char* q, int count1, int count2, int v)
{
    if (count1 + count2 == 0) {
        cout << p << endl;
        sum++;
        return;
    }

    if (v >= 0 && count1 > 0) {
        *q = '(';
        test2(p, q + 1, count1 - 1, count2, v + 1);
    }

    if (count2 > 0) {
        *q = ')';
        test2(p, q + 1, count1, count2 - 1, v - 1);
    }
}

void train(int batch_count, int per_batch_iter_count, int per_iter_train_count)
{
    time_t t1 = time(NULL);
    srand((int)time(0));
    db::DB* mydb = db::GetDB("lmdb");
    mydb->Open("/home/chendejia/workspace/github/dong/data/mnist_train_lmdb", db::READ);
    db::Cursor* cursor = mydb->NewCursor();
    cursor->SeekToFirst();
    int channels = 1;
    int width = 28;
    int height = 28;
    boost::shared_ptr<Neuron[]> inputImage(new Neuron[height * width]);
    boost::shared_ptr<Data> inputData(new Data(1, channels, height, width));
    inputData->setUp(inputImage);
    //L1.inputLayer
    boost::shared_ptr<InputLayer> inputLayer(new InputLayer());
    inputLayer->setUp(inputData);
    //L2.convLayer1
    boost::shared_ptr<ConvLayer> convLayer1(new ConvLayer());
    convLayer1->init(20, 5, 5);
    convLayer1->setUp(inputLayer->getTopData());
    //L3.poolLayer
    boost::shared_ptr<PoolLayer> poolLayer1(new PoolLayer());
    poolLayer1->init(2, 2, 2, 2);
    poolLayer1->setUp(convLayer1->getTopData());
    //L4.convLayer
    boost::shared_ptr<ConvLayer> convLayer2(new ConvLayer());
    convLayer2->init(50, 5, 5);
    convLayer2->setUp(poolLayer1->getTopData());
    //L5.poolLayer
    boost::shared_ptr<PoolLayer> poolLayer2(new PoolLayer());
    poolLayer2->init(2, 2, 2, 2);
    poolLayer2->setUp(convLayer2->getTopData());
    //L6.FullConnectLayer
    boost::shared_ptr<FullConnectLayer> fullConnectLayer(new FullConnectLayer());
    fullConnectLayer->init(500);
    fullConnectLayer->setUp(poolLayer2->getTopData());
    //L7.reluLayer
    boost::shared_ptr<ReluLayer> reluLayer(new ReluLayer());
    reluLayer->init();
    reluLayer->setUp(fullConnectLayer->getTopData());
    //L8.FullConnectLayer
    boost::shared_ptr<FullConnectLayer> fullConnectLayer2(new FullConnectLayer());
    fullConnectLayer2->init(10);
    fullConnectLayer2->setUp(reluLayer->getTopData());
    //L9.SoftmaxLayer
    boost::shared_ptr<SoftmaxLayer> softmaxLayer(new SoftmaxLayer(dong::TRAIN));
    softmaxLayer->init();
    softmaxLayer->setUp(fullConnectLayer2->getTopData());
    float loss_record_sum = 0.0F;
    int record_count = 0;
    Data batchDatas(per_iter_train_count, 1, height, width, Data::CONSTANT);
    int batchLabels[per_iter_train_count];

    //训练batch_count批数据
    for (int batch = 0; batch < batch_count; ++batch) {
        cout << "batch: " << batch << endl;

        //读取一批数据
        for (int i = 0; i < per_iter_train_count && cursor->valid(); i++, cursor->Next()) {
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

                convLayer1->forward();
                poolLayer1->forward();
                convLayer2->forward();
                poolLayer2->forward();
                fullConnectLayer->forward();
                reluLayer->forward();
                fullConnectLayer2->forward();
                softmaxLayer->setLabel(label);
                softmaxLayer->forward();
                softmaxLayer->backward();
                fullConnectLayer2->backward();
                reluLayer->backward();
                fullConnectLayer->backward();
                poolLayer2->backward();
                convLayer2->backward();
                poolLayer1->backward();
                convLayer1->backward();
                ++record_count;
                loss_record_sum += softmaxLayer->getLoss();
            }

            float avg_loss = loss_record_sum / record_count;
            cout << "avg loss:" << setprecision(8) << fixed << avg_loss << endl;
            loss_record_sum = 0.0F;
            record_count = 0;
        }
    }

    cout << "---------convLayer1 weight-----------" << endl;
    convLayer1->getWeightData()->print();
    convLayer1->getTopData()->genBmp("convLayer1_top_data_%d_%d.bmp", 1);
    convLayer1->getWeightData()->genBmp("convLayer1_Weight_data_%d_%d.bmp", 1);
    cout << "---------convLayer2 weight-----------" << endl;
    convLayer2->getWeightData()->print();
    convLayer2->getTopData()->genBmp("convLayer2_top_data_%d_%d.bmp", 1);
    convLayer2->getWeightData()->genBmp("convLayer2_Weight_data_%d_%d.bmp", 1);
    cout << "---------convLayer2 _bias-----------" << endl;
    convLayer2->getBiasData()->print();
    delete cursor;
    mydb->Close();
    delete mydb;
    time_t t2 = time(NULL);
    cout << "训练速度:" << batch_count* per_batch_iter_count* per_iter_train_count /
         (t2 - t1 + 1) << " pic / s" << endl;
}

int main(int argc, char* argv[])
{
    int batch_count = 1;
    int per_batch_iter_count = 1;
    int per_iter_train_count = 1;
    Layer::BASE_LEARNING_RATE = 0.0001;

    if (argc == 5) {
        batch_count = atoi(argv[1]);
        per_batch_iter_count = atoi(argv[2]);
        per_iter_train_count = atoi(argv[3]);
        Layer::BASE_LEARNING_RATE = atof(argv[4]);
    }

    train(batch_count, per_batch_iter_count, per_iter_train_count);
    //threadTest();
    cout << "Hello world!" << endl;
    return 0;
}
