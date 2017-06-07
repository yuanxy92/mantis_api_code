mkdir ./build/
mkdir -p ./bin/Release
cd ./build/
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
#export CC=/usr/bin/clang
#export CXX=/usr/bin/clang++
cmake -DCMAKE_BUILD_TYPE=Release -DEXECUTABLE_OUTPUT_PATH=../bin/Release ..