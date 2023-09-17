# FDCExample
[PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy)的capi简易封装example，使用该例子可以使用其它语言进行调用，其它模型同理
模型文件[百度网盘](https://pan.baidu.com/s/1LPziznc-VFPmjmaxYpkiIw?pwd=f0kf)下载并解压在根目录

#### 1.环境准备
##### 1.1 FastDeploy安装
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
- vcpkg安装
```shell
vcpkg install jsoncpp
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
todo
#### 2. 编译安装
```shell
camke ..
cmake --build . --config Release -j12
```

