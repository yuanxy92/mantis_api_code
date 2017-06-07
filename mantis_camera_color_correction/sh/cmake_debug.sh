mkdir ./build/
mkdir -p ./bin/Debug
cd ./build/
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
#export CC=/usr/bin/clang
#export CXX=/usr/bin/clang++
cmake -DCMAKE_BUILD_TYPE=Debug -DEXECUTABLE_OUTPUT_PATH=../bin/Debug ..
