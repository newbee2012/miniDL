#include "common.hpp"
#include "data.hpp"
#include "util/gen_bmp.hpp"
#include <boost/shared_ptr.hpp>
#include <vector>
#include <iomanip>
#include "util/math_utils.hpp"

namespace dong
{
Data::Data(int num, int channels, int height, int width): _num(num), _channels(channels), _height(height),
    _width(width) {};


Data::Data(int num, int channels, int height, int width, boost::shared_ptr<InitDataParam>& param): _num(num), _channels(channels),
    _height(height), _width(width)
{
    _neurons.reset(new Neuron[count()]);
    int fan_in = count() / num;
    int fan_out = count() / channels;
    //float n = (fan_in + fan_out) / float(2);
    float n = fan_in;
    float scale = sqrt(float(3) / n);
    float* t = new float[count()];
    if (param->_initType == XAVIER)
    {
        RandomGenerator::rng_uniform(count(), -scale, scale, t);
    }
    else if(param->_initType == GAUSSIAN)
    {
        RandomGenerator::rng_gaussian(count(), param->_gaussian_mean, param->_gaussian_std, t);
    }

    for (int i = 0; i < count(); ++i)
    {
        if (param->_initType == CONSTANT)
        {
            _neurons[i]._value = param->_constant_value;
        }
        else if (param->_initType == RANDOM)
        {
            _neurons[i]._value = (float)random(2) - 0.5F;
            _neurons[i]._value /= 1000;
        }else
        {
            _neurons[i]._value = t[i];
        }

        _neurons[i]._diff = 0.0F;
    }

    delete[] t;
}

void Data::clearDiff()
{
    int c = count();
    for (int i = 0; i < c; ++i)
    {
        _neurons[i]._diff = 0.0F;
    }
}

void Data::clearValue()
{
    int c = count();
    for (int i = 0; i < c; ++i)
    {
        _neurons[i]._value = 0.0F;
    }
}

void Data::setUp(const boost::shared_array<Neuron>& neurons)
{
    _neurons = neurons;
}

void Data::print()
{
    cout  << "h:" << _height << ",w:" << _width << endl;

    for (int n = 0; n < _num; n++)
    {
        for (int c = 0; c < _channels; c++)
        {
            for (int h = 0; h < _height; h++)
            {
                for (int w = 0; w < _width; w++)
                {
                    float value = this->get(n, c, h, w)->_value;
                    cout << setprecision(3)<<fixed<< value;
                    if (value < 10)
                    {
                        cout << "   ";
                    }
                    else if (value < 100)
                    {
                        cout << "  ";
                    }
                    else
                    {
                        cout << " ";
                    }
                }

                if (_width > 1)
                {
                    cout << endl << endl;
                }
            }

        }
    }
}

void Data::printDiff()
{
    for (int n = 0; n < _num; n++)
    {
        for (int c = 0; c < _channels; c++)
        {
            for (int h = 0; h < _height; h++)
            {
                for (int w = 0; w < _width; w++)
                {
                    float value = this->get(n, c, h, w)->_diff;
                    cout << value;

                    //cout << setprecision(2)<<fixed<< value;
                    if (value < 10)
                    {
                        cout << "   ";
                    }
                    else if (value < 100)
                    {
                        cout << "  ";
                    }
                    else
                    {
                        cout << " ";
                    }
                }

                if (_width > 1)
                {
                    cout << endl << endl;
                }
            }

            cout << "----------------------------------" << endl;
        }
    }
}
/*
void Data::genBmp(string& filePathBase)
{
    filePathBase.append("_%d_%d.bmp");
    for (int n = 0; n < _num; n++) {
        for (int c = 0; c < _channels; c++) {
            char filePath[128]={0};
            sprintf(filePath,filePathBase.c_str(),n,c);
            cout<<"genBmp : " << filePath<<endl;
            RGB* pRGB = new RGB[_width * _height];
            memset(pRGB, 0x0, sizeof(RGB) * _width * _height); // 设置背景为黑色

            for (int h = 0; h < _height; h++) {
                for (int w = 0; w < _width; w++) {
                    BYTE gray = this->get(n, c, h, w)->_value > 1 ? this->get(n, c, h, w)->_value : (this->get(n, c, h,
                                w)->_value) * 0xFF;
                    if (w == 0 || w == _width - 1 || h == 0 || h == _height - 1) {
                        pRGB[(_height - h - 1)*_width + w].r = 0xFF;
                        pRGB[(_height - h - 1)*_width + w].g = 0xFF;
                        pRGB[(_height - h - 1)*_width + w].b = 0xFF;
                    } else {
                        pRGB[(_height - h - 1)*_width + w].r = gray;
                        pRGB[(_height - h - 1)*_width + w].g = gray;
                        pRGB[(_height - h - 1)*_width + w].b = gray;
                    }
                }
            }

            BmpTool::generateBMP((BYTE*)pRGB, _width, _height, filePath);
        }
    }
}*/

void Data::genBmp(string& filePathBase)
{
    filePathBase.append("_%d_%d.bmp");
    RGB* pRGB = new RGB[_width * _height];
    for (int n = 0; n < _num; n++)
    {
        for (int c = 0; c < _channels; c++)
        {
            char filePath[128]= {0};
            sprintf(filePath,filePathBase.c_str(),n,c);
            cout<<"genBmp : " << filePath<<endl;
            memset(pRGB, 0x0, sizeof(RGB) * _width * _height); // 设置背景为黑色
            for (int h = 0; h < _height; h++)
            {
                for (int w = 0; w < _width; w++)
                {
                    /*
                    if (w == 0 || w == _width - 1 || h == 0 || h == _height - 1)
                    {
                        pRGB[(_height - h - 1)*_width + w].r = 0xFF;
                        pRGB[(_height - h - 1)*_width + w].g = 0xFF;
                        pRGB[(_height - h - 1)*_width + w].b = 0xFF;
                    }
                    else
                    {
                        if(this->_channels == 1)
                        {
                            pRGB[(_height - h - 1)*_width + w].r = this->get(n, 0, h, w)->_value;
                            pRGB[(_height - h - 1)*_width + w].g = this->get(n, 0, h, w)->_value;
                            pRGB[(_height - h - 1)*_width + w].b = this->get(n, 0, h, w)->_value;
                        }
                        else if(this->_channels == 3)
                        {
                            pRGB[(_height - h - 1)*_width + w].r = this->get(n, 0, h, w)->_value;
                            pRGB[(_height - h - 1)*_width + w].g = this->get(n, 1, h, w)->_value;
                            pRGB[(_height - h - 1)*_width + w].b = this->get(n, 2, h, w)->_value;
                        }
                    }*/

                    pRGB[(_height - h - 1)*_width + w].r = this->get(n, c, h, w)->_value;
                    pRGB[(_height - h - 1)*_width + w].g = this->get(n, c, h, w)->_value;
                    pRGB[(_height - h - 1)*_width + w].b = this->get(n, c, h, w)->_value;

                }
            }
            BmpTool::generateBMP(pRGB, _width, _height, filePath);
        }
    }

    delete[] pRGB;
}


}  // namespace dong

