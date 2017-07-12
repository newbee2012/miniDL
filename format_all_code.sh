./astyle -A10 --recursive --line-ends-mixed ./*.hpp ./*.cpp ./*.h ./*.c
./astyle -A10 -xW -Y -f -p -xg -H -xe -k1 -W1 -j -xC120 -n --mode=c --recursive ./*.hpp ./*.cpp ./*.h ./*.c

