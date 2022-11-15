//include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//include <opencv2/core/core.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include<inttypes.h>
#include "common.hpp"
#include "util/gen_bmp.hpp"
using namespace std;
//using namespace cv;

namespace dong
{

    class MnistUtils
    {
        public:

        //把大端数据转换为我们常用的小端数据
        static uint32_t swap_endian(uint32_t val)
        {
            val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
            return (val << 16) | (val >> 16);
        }

        static void readAndSave(const string& mnist_img_path, const string& mnist_label_path)
        {
            //以二进制格式读取mnist数据库中的图像文件和标签文件
            ifstream mnist_image(mnist_img_path, ios::in | ios::binary);
            ifstream mnist_label(mnist_label_path, ios::in | ios::binary);
            if (mnist_image.is_open() == false)
            {
                cout << "open mnist image file error!" << endl;
                return;
            }
            if (mnist_label.is_open() == false)
            {
                cout << "open mnist label file error!" << endl;
                return;
            }

            uint32_t magic;//文件中的魔术数(magic number)
            uint32_t num_items;//mnist图像集文件中的图像数目
            uint32_t num_label;//mnist标签集文件中的标签数目
            uint32_t rows;//图像的行数
            uint32_t cols;//图像的列数
            //读魔术数
            mnist_image.read(reinterpret_cast<char*>(&magic), 4);
            magic = swap_endian(magic);
            if (magic != 2051)
            {
                cout << "this is not the mnist image file" << endl;
                return;
            }
            mnist_label.read(reinterpret_cast<char*>(&magic), 4);
            magic = swap_endian(magic);
            if (magic != 2049)
            {
                cout << "this is not the mnist label file" << endl;
                return;
            }
            //读图像/标签数
            mnist_image.read(reinterpret_cast<char*>(&num_items), 4);
            num_items = swap_endian(num_items);
            mnist_label.read(reinterpret_cast<char*>(&num_label), 4);
            num_label = swap_endian(num_label);
            //判断两种标签数是否相等
            if (num_items != num_label)
            {
                cout << "the image file and label file are not a pair" << endl;
            }
            //读图像行数、列数
            mnist_image.read(reinterpret_cast<char*>(&rows), 4);
            rows = swap_endian(rows);
            mnist_image.read(reinterpret_cast<char*>(&cols), 4);
            cols = swap_endian(cols);
            //读取图像
            for (int i = 0; i != num_items; i++)
            {
                char* pixels = new char[rows * cols];
                mnist_image.read(pixels, rows * cols);
                char label;
                mnist_label.read(&label, 1);
                string path="./pic/";
                path.append(to_string(i)).append("_").append(to_string((unsigned int)label)).append(".jpg");
                BmpTool::generateBMP((BYTE*)pixels,rows,cols,path.c_str());
            }
        }
    };
}

