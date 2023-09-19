//
// Created by DELL on 2023/9/19.
//

#include "defect_detect.h"
#include <iostream>
#include <opencv2/opencv.hpp>


void test_cls() {
    model_handle_t model = nullptr;
    ClsResult result;
    init_model(&model, "../models/inference_model_cls/", ModeType::REC_MODEL, 2, true);
    auto img = cv::imread("../demos/93.jpg");
    shape_classify(model, img.data, img.cols, img.rows, &result);
    //{0:"circle", 1:"square", 2:"triangle"}
    for (int i = 0; i < result.size; i++) {
        std::cout << "ids:" << result.ids[i] << std::endl;
        std::cout << "score:" << result.scores[i] << std::endl;
    }
    free_model(model, ModeType::REC_MODEL);
}

void test_object_det_chr() {
    model_handle_t model = nullptr;
    init_model(&model, "../models/inference_model_qx_det/", ModeType::DET_MODEL, 2, true);
    char *ret = (char *) malloc(10000);
    obj_detection_str(model, "../demos/22.bmp", 0.75, ret);
    std::cout << ret << std::endl;
    free_ret_result(ret);
    free_model(model, ModeType::DET_MODEL);
}

int main() {
//    test_cls();
    test_object_det_chr();
}