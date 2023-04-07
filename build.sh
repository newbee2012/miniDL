#!/bin/sh
set -v on

g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/conv_layer.cpp -o obj/Release/src/conv_layer.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/data.cpp -o obj/Release/src/data.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/full_connect_layer.cpp -o obj/Release/src/full_connect_layer.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/input_layer.cpp -o obj/Release/src/input_layer.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/layer.cpp -o obj/Release/src/layer.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/main.cpp -o obj/Release/src/main.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/net_model.cpp -o obj/Release/src/net_model.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/net_model_binary.cpp -o obj/Release/src/net_model_binary.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/neuron.cpp -o obj/Release/src/neuron.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/pool_layer.cpp -o obj/Release/src/pool_layer.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/relu_layer.cpp -o obj/Release/src/relu_layer.o
g++ -Winline -Wfatal-errors -fexceptions -O3 -Iinclude -Iextern/include -c /home/chendejia/workspace/github/miniDL/src/softmax_layer.cpp -o obj/Release/src/softmax_layer.o
mkdir -p bin/Release
g++ -Lextern/lib -o bin/Release/miniDL obj/Release/src/conv_layer.o obj/Release/src/data.o obj/Release/src/full_connect_layer.o obj/Release/src/input_layer.o obj/Release/src/layer.o obj/Release/src/main.o obj/Release/src/net_model.o obj/Release/src/net_model_binary.o obj/Release/src/neuron.o obj/Release/src/pool_layer.o obj/Release/src/relu_layer.o obj/Release/src/softmax_layer.o  -O3 -static-libstdc++ -static -s  -lboost_system -lpthread -ljsoncpp
