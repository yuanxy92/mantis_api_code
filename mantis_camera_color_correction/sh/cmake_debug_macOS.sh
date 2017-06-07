mkdir ./build/
mkdir -p ./bin/Debug
cd ./build/
cmake -DCMAKE_BUILD_TYPE=Debug -DEXECUTABLE_OUTPUT_PATH=../bin/Debug ..
