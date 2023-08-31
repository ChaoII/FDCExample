#include "defect_detect.h"
#include <opencv2/opencv.hpp>

void test_det();

void test_cls();

int main() {
    test_det();
//    test_cls();
}

void test_det() {
    void *model = nullptr;
    DetResult result;
    init_model(&model, "../inference_model_shape_new/", ModeType::DET_MODEL, 2, true);
    auto img = cv::imread("../00020.jpg");
    void *buffer = malloc(img.total() * img.elemSize());
    //{0:"square", 1:"triangle",2:"circle"}
    obj_detection(model, img.data, buffer, img.cols, img.rows, &result, 0.5, true);
    cv::Mat dst_img = cv::Mat(img.rows, img.cols, CV_8UC3, buffer);
    cv::imshow("img", dst_img);
    cv::waitKey(0);
    free_model(model, ModeType::DET_MODEL);
}

void test_cls() {
    void *model = nullptr;
    ClsResult result;
    init_model(&model, "../inference_model_cls/", ModeType::REC_MODEL, 2, true);
    auto img = cv::imread("../93.jpg");
    shape_classify(model, img.data, img.cols, img.rows, &result);
    //{0:"circle", 1:"square", 2:"triangle"}
    for (int i = 0; i < result.size; i++) {
        std::cout << "ids:" << result.ids[i] << std::endl;
        std::cout << "score:" << result.scores[i] << std::endl;
    }
    free_model(model, ModeType::REC_MODEL);
}







