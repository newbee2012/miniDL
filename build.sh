#!/bin/sh
set -v on

g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/conv_layer.cpp -o obj/Release/src/conv_layer.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/data.cpp -o obj/Release/src/data.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/full_connect_layer.cpp -o obj/Release/src/full_connect_layer.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/input_layer.cpp -o obj/Release/src/input_layer.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/layer.cpp -o obj/Release/src/layer.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/main.cpp -o obj/Release/src/main.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/net_model.cpp -o obj/Release/src/net_model.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/net_model_lmdb.cpp -o obj/Release/src/net_model_lmdb.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/net_model_mysql.cpp -o obj/Release/src/net_model_mysql.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/neuron.cpp -o obj/Release/src/neuron.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/pool_layer.cpp -o obj/Release/src/pool_layer.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/proto/caffe.pb.cc -o obj/Release/src/proto/caffe.pb.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/relu_layer.cpp -o obj/Release/src/relu_layer.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/softmax_layer.cpp -o obj/Release/src/softmax_layer.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/util/db.cpp -o obj/Release/src/util/db.o
g++ -Winline -Wfatal-errors -fexceptions -DUSE_LMDB -O3 -Iinclude -I/usr/include/jsoncpp -c /home/chendejia/workspace/github/miniDL/src/util/db_lmdb.cpp -o obj/Release/src/util/db_lmdb.o
g++  -o bin/Release/miniDL obj/Release/src/conv_layer.o obj/Release/src/data.o obj/Release/src/full_connect_layer.o obj/Release/src/input_layer.o obj/Release/src/layer.o obj/Release/src/main.o obj/Release/src/net_model.o obj/Release/src/net_model_lmdb.o obj/Release/src/net_model_mysql.o obj/Release/src/neuron.o obj/Release/src/pool_layer.o obj/Release/src/proto/caffe.pb.o obj/Release/src/relu_layer.o obj/Release/src/softmax_layer.o obj/Release/src/util/db.o obj/Release/src/util/db_lmdb.o  -s  -lprotobuf -lglog -llmdb -lgflags -lboost_system -lpthread -lmysqlclient -ljsoncpp -lcblas
