cmake_minimum_required(VERSION 3.20)

project(defect_detection)

set(FASTDEPLOY_DIR "E:/FastDeploy/build/install")

set(DLL_NAME "fdd")
include(${FASTDEPLOY_DIR}/FastDeploy.cmake)

include_directories(${FASTDEPLOY_INCS})

set(CMAKE_CXX_STANDARD 17)

add_library(${DLL_NAME} SHARED
        defect_detect.h
        defect_detect.cpp)


target_link_libraries(${DLL_NAME} ${FASTDEPLOY_LIBS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${DLL_NAME})
