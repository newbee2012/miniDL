#include <iostream>
#include "util/math_utils.hpp"
#include "util/mnist_utils.hpp"
#include "net_model_binary.hpp"
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <getopt.h>
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
boost::shared_ptr<InitDataParam> Layer::default_init_data_param;
boost::shared_ptr<rng_t> RandomGenerator::engine;

void runModel(string modelFilePath)
{
    NetModelBinary* netMode = new NetModelBinary(modelFilePath);
    netMode->load_model();
    netMode->run();
    //netMode->compute_mean();
    delete netMode;
}

void forecastBmp(string modelFilePath, string picFilePath)
{
    NetModelBinary* netMode = new NetModelBinary(modelFilePath);
    netMode->load_model();
    netMode->testFromABmp(picFilePath);
    delete netMode;
}

void display_help() {
    std::cout << "Usage: miniDL -m {model_define_file_path} [-i {test_image_file_path}] [-r {rnd_seed}]\n\n";
    std::cout << "  -m, --model_define: path to the model definition file (required)\n";
    std::cout << "  -i, --test_image_path: path to the test image file (optional)\n";
    std::cout << "  -r, --rnd_seed: random seed for initialization (optional)\n";
    std::cout << "  --help: display this help message\n";
}

int main(int argc, char* argv[])
{
std::string model_define_file_path, test_image_file_path;
    int rnd_seed = -1;
    int c;

    if (argc < 2) {
        display_help();
        return 1;
    }

    while (true) {
        int option_index = 0;
        static struct option long_options[] = {
            {"model_define", required_argument, 0, 'm'},
            {"test_image_path", optional_argument, 0, 'i'},
            {"rnd_seed", optional_argument, 0, 'r'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };

        c = getopt_long(argc, argv, "m:i:r:h", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
        case 'm':
            model_define_file_path = optarg;
            break;

        case 'i':
            test_image_file_path = optarg;
            break;

        case 'r':
            rnd_seed = atoi(optarg);
            break;

        case 'h':
        case '?':
        default:
            display_help();
            return 1;
        }
    }

    if (model_define_file_path.empty()) {
        std::cerr << "Error: Model definition file path is required!\n";
        display_help();
        return 1;
    }


    // Initialize random seed (if provided)
    RandomGenerator::init_engine(rnd_seed);

    std::cout << "Model define file path: " << model_define_file_path << std::endl;

    // Process model definition file and test image file (if provided)
    if (test_image_file_path != "") {
        std::cout << "Test image file path: " << test_image_file_path << std::endl;
        forecastBmp(model_define_file_path, test_image_file_path);
    }else
    {
        runModel(model_define_file_path);
    }

    return 0;
}

struct Node
{
    Node* next;
};
