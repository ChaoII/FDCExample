cmake_minimum_required(VERSION 3.20)

project(defect_detection)

set(CMAKE_CXX_STANDARD 17)

set(FASTDEPLOY_DIR "E:/FastDeploy/build_cpu/install" CACHE PATH "fastdeploy install directory")
set(gflags_DIR "F:/gflags/bd/install/lib/cmake/gflags" CACHE PATH "gflags cmake directory")
set(jsoncpp_DIR "F:/jsoncpp/build/install/lib/cmake/jsoncpp" CACHE PATH "jsoncpp cmake directory")
set(DLL_NAME "FDD" CACHE STRING "export dll name")
message("-------------------")
message("${gflags_DIR}")
message("${jsoncpp_DIR}")

include(${FASTDEPLOY_DIR}/FastDeploy.cmake)
include_directories(${FASTDEPLOY_INCS})
find_package(gflags REQUIRED)
find_package(jsoncpp REQUIRED)

add_library(${DLL_NAME} SHARED
        defect_detect.h
        defect_detect.cpp)

target_link_libraries(${DLL_NAME} ${FASTDEPLOY_LIBS})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${DLL_NAME} JsonCpp::JsonCpp ${GFLAGS_LIBRARIES})
