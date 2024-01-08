# FDCExample
[PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy)的capi简易封装example，使用该例子可以使用其它语言进行调用，其它模型同理
模型文件[百度网盘](https://pan.baidu.com/s/1LPziznc-VFPmjmaxYpkiIw?pwd=f0kf)下载并解压在根目录

#### 1.环境准备
##### 1.1 FastDeploy安装
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -G Ninja -DENABLE_VISION=ON -DENABLE_ORT_BEAKEND=ON -DOPENCV_DIRECTORY="your opencv directory" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install
cmake --build . --config Relase -j20
cmake --build . --config Relase --target install
```
##### 1.2 jsoncpp安装
###### 1.2.1 windows
- 源码编译
```shell
git clone https://github.com/open-source-parsers/jsoncpp.git
cd jsoncpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=install
cmake --build . --config Release -j12
cmake --build . --config Release --target install
```
###### 1.2.2 linux（ubuntu为例）
- 源码编译
```shell
git clone https://github.com/open-source-parsers/jsoncpp.git
cd jsoncpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=install
make -j4
sudo make install
```
- 直接安装
```shell
sudo apt install libjsoncpp-dev
```
##### 1.3 安装gflag
```shell
git clone  https://github.com/gflags/gflags.git
cd gflags
mkdir bd && cd bd
cmake .. -DCMAKE_BUILD_PREFIX=install
cmake --build . --config Release -j12
cmake --build . --config Release --target install
```
#### 2. 编译安装
```shell
git clone https://github.com/ChaoII/FDCExample.git
cd FDCExample
git submodule update --init
mkdir build && cd build
# 其中FASTDEPLOY_DIR为第一步源码编译FastDeploy得到的install路径
# DUSE_SUBMODULE可以不依赖于本地编译gflags和jsoncpp
camke .. -DFASTDEPLOY_DIR="Fastdeploy directory" -DBUILD_TESTING=fasle -DUSE_SUBMODULE=ON
cmake --build . --config Release -j12
```

