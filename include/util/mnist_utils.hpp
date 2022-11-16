#ifndef MNIST_UTILS_HPP_
#define MNIST_UTILS_HPP_
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include<inttypes.h>
#include "common.hpp"
#include "util/gen_bmp.hpp"
using namespace std;

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

        static void readData(const string& mnist_img_path, const string& mnist_label_path, boost::shared_array<char>& pPixels, size_t& pixels_size, boost::shared_array<char>& pLabels, size_t& labels_size)
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
            pixels_size = num_items * rows * cols;
            labels_size = num_label;
            pPixels.reset(new char[pixels_size]);
            pLabels.reset(new char[labels_size]);
            //读取图像
            mnist_image.read(pPixels.get(), pixels_size);
            mnist_label.read(pLabels.get(), labels_size);
        }

        static void convertToBMP(const string& mnist_img_path, const string& mnist_label_path)
        {
            boost::shared_array<char> pPixels;
            size_t pixels_size=0;
            boost::shared_array<char> pLabels;
            size_t labels_size=0;

            readData(mnist_img_path,mnist_label_path,pPixels,pixels_size,pLabels,labels_size);

            for(int i=0;i<10;i++){
                string path="./pic/";
                char label = pLabels[i];
                path.append(to_string(i)).append("_").append(to_string((unsigned int)label)).append(".bmp");
                int offset = i * 28 * 28;
                RGB rgb[28][28];

                for(int row=0; row < 28; row++)
                {
                    for(int col=0; col < 28; col++)
                    {
                        BYTE value = (BYTE)pPixels[offset + (28-row) * 28 + col];
                        rgb[row][col].r = value;
                        rgb[row][col].g = value;
                        rgb[row][col].b = value;
                    }
                }
                cout<<"generateBMP:"<<path.c_str()<<endl;
                BmpTool::generateBMP(&rgb[0][0],28,28,path.c_str());
            }
        }
    };
}
#endif
