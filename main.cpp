#include "defect_detect.h"

#include <gflags/gflags.h>
#include <json/json.h>
#include <iostream>
#include <opencv2/opencv.hpp>

DEFINE_string(config_file, "config.json","config file");

static std::unordered_map<int, std::string> index_map{std::make_pair<int, std::string>(0, "square"),
                                                      std::make_pair<int, std::string>(1, "triangle"),
                                                      std::make_pair<int, std::string>(2, "circle")};
void test_cls();
/* the location index is like blew
 *|-----------|----------|
 *|           |          |
 *|     0     |     1    |
 *|           |          |
 *|-----------|----------|
 *|           |          |
 *|     2     |     3    |
 *|           |          |
 *------------|----------|
 * */
void execute_det(const char *image_path,
                 const char *model_dir,
                 const ModeType &model_type,
                 DetResult *result,
                 bool show_image,
                 bool save_image,
                 int thread_num,
                 bool use_gpu) {
    void *model = nullptr;
    init_model(&model, model_dir, model_type, thread_num, use_gpu);
    auto img = cv::imread(image_path);
    cv::resize(img, img, cv::Size(0, 0), 0.4, 0.4);
    void *buffer = malloc(img.total() * img.elemSize());
    //{0:"square", 1:"triangle",2:"circle"}
    obj_detection(model, img.data, buffer, img.cols, img.rows, result, 0.5, true);
    std::cout << "size is: " << result->size << std::endl;
    cv::Mat dst_img = cv::Mat(img.rows, img.cols, CV_8UC3, buffer);
    if (show_image) {
        cv::imshow("img", dst_img);
        cv::waitKey(0);
    }
    if (save_image) {
        std::cout << "image save to [result.jpg] successfully" << std::endl;
        cv::imwrite("result.jpg", dst_img);
    }
    free_model(model, ModeType::DET_MODEL);
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
//    if (argc < 2) {
//        std::cerr << "[error]please add config.json parameter" << std::endl;
//        return -1;
//    }
    Json::Value root;
    std::ifstream ifs;
    DetResult result;
    ifs.open(FLAGS_config_file);
    Json::CharReaderBuilder reader_builder;
    JSONCPP_STRING errs;
    if (!parseFromStream(reader_builder, ifs, &root, &errs)) {
        std::cout << errs << std::endl;
        return -1;
    }
    std::cout << "[info]input config is:" << std::endl;
    std::cout << root << std::endl;
    std::string image_path = root.get("image_path", "").asString();
    std::string model_dir = root.get("model_dir", "").asString();
    int model_type = root.get("model_type", 1).asInt();
    int thread_num = root.get("thread_num", 1).asInt();
    bool use_gpu = root.get("use_gpu", false).asBool();
    bool show_image = root.get("show_image", false).asBool();
    bool save_image = root.get("save_image", false).asBool();
    execute_det(image_path.c_str(), model_dir.c_str(),
                (ModeType) model_type,
                &result, show_image, save_image, thread_num, use_gpu);

    Json::Value dst_root, sub, box_sub;
    for (size_t i = 0; i < result.size; i++) {
        box_sub["x_min"] = result.box[i].x_min;
        box_sub["x_max"] = result.box[i].x_max;
        box_sub["y_min"] = result.box[i].y_min;
        box_sub["y_max"] = result.box[i].y_max;
        sub["box"] = box_sub;
        sub["score"] = result.scores[i];
        sub["label"] = index_map[result.label_ids[i]];
        sub["location"] = result.local_index[i];
        dst_root.append(sub);
    }
    std::cout << "[info] result is:" << std::endl;
    std::cout << dst_root << std::endl;
    Json::StreamWriterBuilder writer_builder;
    const std::unique_ptr<Json::StreamWriter> writer(writer_builder.newStreamWriter());
    std::ofstream write_ofs("result.json");
    write_ofs << Json::writeString(writer_builder, dst_root);
    write_ofs.close();
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