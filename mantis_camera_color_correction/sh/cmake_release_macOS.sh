mkdir ./build/
mkdir -p ./bin/Release
cd ./build/
cmake -DCMAKE_BUILD_TYPE=Release -DEXECUTABLE_OUTPUT_PATH=../bin/Release ..
