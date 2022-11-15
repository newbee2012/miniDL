#ifndef DONG_GEN_BMP_HPP_
#define DONG_GEN_BMP_HPP_
#include "common.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

namespace dong
{
#pragma pack(push, 2)//必须得写，否则sizeof得不到正确的结果
typedef struct
{
    BYTE b;
    BYTE g;
    BYTE r;
} RGB;

typedef struct
{
    WORD    bfType;
    DWORD   bfSize;
    WORD    bfReserved1;
    WORD    bfReserved2;
    DWORD   bfOffBits;
} BITMAPFILEHEADER;

typedef struct
{
    DWORD      biSize;
    INT32       biWidth;
    INT32       biHeight;
    WORD       biPlanes;
    WORD       biBitCount;
    DWORD      biCompression;
    DWORD      biSizeImage;
    INT32       biXPelsPerMeter;
    INT32       biYPelsPerMeter;
    DWORD      biClrUsed;
    DWORD      biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

class BmpTool
{
public:
    static BYTE* readBmp(const char *bmpName)
    {
        FILE *fp;
        if( (fp = fopen(bmpName,"rb")) == NULL)  //以二进制的方式打开文件
        {
            cout<<"The file "<<bmpName<<"was not opened"<<endl;
            return NULL;
        }
        if(fseek(fp,sizeof(BITMAPFILEHEADER),0))  //跳过BITMAPFILEHEADE
        {
            cout<<"跳转失败"<<endl;
            return NULL;
        }
        BITMAPINFOHEADER infoHead;
        fread(&infoHead,sizeof(BITMAPINFOHEADER),1,fp);   //从fp中读取BITMAPINFOHEADER信息到infoHead中,同时fp的指针移动
        int bmpwidth = infoHead.biWidth;
        int bmpheight = infoHead.biHeight;
        int linebyte = (bmpwidth*24/8+3)/4*4; //计算每行的字节数，24：该图片是24位的bmp图，3：确保不丢失像素

        BYTE* pBmpBuf = new BYTE[linebyte*bmpheight];
        fread(pBmpBuf,sizeof(char),linebyte*bmpheight,fp);
        fclose(fp);   //关闭文件
        return pBmpBuf;
    }

    static void generateBMP(RGB* pData, int width, int height, const char* filename)
    {
        int size = width * height * sizeof(RGB); // 每个像素点3个字节
        // 位图第一部分，文件信息
        BITMAPFILEHEADER bfh;
        bfh.bfType = 0x4d42;//bm
        bfh.bfSize = size  // data size
                     + sizeof( BITMAPFILEHEADER ) // first section size
                     + sizeof( BITMAPINFOHEADER ) // second section size
                     ;
        bfh.bfReserved1 = 0; // reserved
        bfh.bfReserved2 = 0; // reserved
        bfh.bfOffBits = bfh.bfSize - size;
        // 位图第二部分，数据信息
        BITMAPINFOHEADER bih;
        bih.biSize = sizeof(BITMAPINFOHEADER);
        bih.biWidth = width;
        bih.biHeight = height;
        bih.biPlanes = 1;
        bih.biBitCount = 24;
        bih.biCompression = 0;
        bih.biSizeImage = size;
        bih.biXPelsPerMeter = 0;
        bih.biYPelsPerMeter = 0;
        bih.biClrUsed = 0;
        bih.biClrImportant = 0;
        FILE* fp = fopen( filename, "wb" );

        if ( !fp )
        {
            return;
        }

        fwrite( &bfh, 1, sizeof(BITMAPFILEHEADER), fp );
        fwrite( &bih, 1, sizeof(BITMAPINFOHEADER), fp );
        fwrite( pData, sizeof(RGB), width * height, fp );
        fclose( fp );
    }

};

}


#endif  // DONG_GEN_BMP_HPP_
