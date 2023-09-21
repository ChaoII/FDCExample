//
// Created by DELL on 2023/6/6.
//

#include "defect_detect.h"
#include <json/json.h>

bool init_det_model(model_handle_t *model_handle, const char *model_dir, const fastdeploy::RuntimeOption &opt) {

    std::string model_path = std::string(model_dir) + "model.pdmodel";
    std::string param_path = std::string(model_dir) + "model.pdiparams";
    std::string config_path = std::string(model_dir) + "infer_cfg.yml";
    *model_handle = new fastdeploy::vision::detection::PPYOLOE(model_path,
                                                               param_path,
                                                               config_path,
                                                               opt);
    if (!static_cast<fastdeploy::vision::detection::PPYOLOE *>(*model_handle)->Initialized()) {
        return false;
    }
    return true;
}

bool init_rec_model(model_handle_t *model_handle, const char *model_dir, const fastdeploy::RuntimeOption &opt) {
    std::string model_path = std::string(model_dir) + "model.pdmodel";
    std::string param_path = std::string(model_dir) + "model.pdiparams";
    std::string config_path = std::string(model_dir) + "infer_cfg.yml";
    *model_handle = new fastdeploy::vision::classification::MobileNetv3(model_path,
                                                                        param_path,
                                                                        config_path,
                                                                        opt);
    if (!static_cast<fastdeploy::vision::classification::MobileNetv3 *>(*model_handle)->Initialized()) {
        return false;
    }
    return true;
}

std::vector<SortedArray> sort_det_result(fastdeploy::vision::DetectionResult &result) {

    int num = (int) result.boxes.size();
    std::vector<SortedArray> sorted_array(num);
    for (int i = 0; i < num; i++) {
        sorted_array[i].index = i;
        sorted_array[i].value = result.scores[i];
    }
    sort(sorted_array.begin(), sorted_array.end(), [](SortedArray a, SortedArray b) {
        return a.value > b.value;
    });
    return sorted_array;
}

bool init_model(model_handle_t *model_handle,
                const char *model_dir,
                ModeType model_type,
                int thread_num,
                bool use_gpu) {
    fastdeploy::RuntimeOption opt;
    opt.UseOrtBackend();
    opt.SetCpuThreadNum(thread_num);
    if (use_gpu) {
        opt.UseGpu();
    }
    if (model_type == ModeType::REC_MODEL) {
        return init_rec_model(model_handle, model_dir, opt);
    } else if (model_type == ModeType::DET_MODEL) {
        return init_det_model(model_handle, model_dir, opt);
    } else {
        return false;
    }
}

bool obj_detection(model_handle_t model_handle,
                   void *buffer,
                   void *out_buffer,
                   int w, int h,
                   DetResult *ret,
                   float vis_threshold,
                   bool draw_text) {
    fastdeploy::vision::DetectionResult result;
    cv::Mat img = cv::Mat(h, w, CV_8UC3, buffer);
    static_cast<fastdeploy::vision::detection::PPYOLOE *>(model_handle)->Predict(img, &result);
    int boxes_num = static_cast<int>(result.boxes.size());
    int num = boxes_num > DET_NUM ? DET_NUM : boxes_num;
    int valid_ret = 0;
    std::vector<SortedArray> sorted_array = sort_det_result(result);
    for (size_t i = 0; i < num; i++) {
        Box box;
        box.x_min = result.boxes[sorted_array[i].index][0];
        box.y_min = result.boxes[sorted_array[i].index][1];
        box.x_max = result.boxes[sorted_array[i].index][2];
        box.y_max = result.boxes[sorted_array[i].index][3];
        ret->box[i] = box;
        ret->scores[i] = result.scores[sorted_array[i].index];
        ret->label_ids[i] = result.label_ids[sorted_array[i].index];
        ret->local_index[i] = get_local_index_by_center_point(box, w, h);
        if (ret->scores[i] > vis_threshold) {
            valid_ret++;
            cv::rectangle(img, cv::Point2f(box.x_min, box.y_min),
                          cv::Point2f(box.x_max, box.y_max),
                          cv::Scalar(0, 255, 255),
                          1,
                          cv::LINE_AA);
            if (draw_text) {
                std::string score_str = std::to_string(ret->scores[i]);
                // 保留4位有效数字
                if (score_str.size() > 4) {
                    score_str = score_str.substr(0, 4);
                }
                std::string text = std::to_string(ret->label_ids[i]) + ", " + score_str;
                int base_line = 0;
                cv::Size text_size = cv::getTextSize(text,
                                                     FONT_FACE,
                                                     FONT_SCALE,
                                                     1,
                                                     &base_line);
                cv::Point2f pt_text_bg1, pt_text_bg2;
                //文本背景框
                pt_text_bg1.x = box.x_min;
                pt_text_bg1.y = box.y_min + (float) text_size.height;
                pt_text_bg2.x = std::max(box.x_min + (float) text_size.width, box.x_max);
                pt_text_bg2.y = box.y_min;
                //文本原点(左下角)
                cv::rectangle(img, pt_text_bg1,
                              pt_text_bg2,
                              cv::Scalar(0, 255, 255),
                              -1, cv::LINE_AA, 0);
                cv::putText(img, text, pt_text_bg1,
                            FONT_FACE,
                            FONT_SCALE,
                            cv::Scalar(255, 0, 0),
                            1,
                            cv::LINE_AA);
            }
        }
    }
    ret->size = valid_ret;
    memcpy(out_buffer, img.data, img.total() * img.elemSize());
    return true;
}


bool shape_classify(model_handle_t model_handle, void *buffer, int w, int h, ClsResult *ret) {
    fastdeploy::vision::ClassifyResult result;
    cv::Mat img = cv::Mat(h, w, CV_8UC3, buffer);
    static_cast<fastdeploy::vision::classification::MobileNetv3 *>(model_handle)->Predict(img, &result);
    int cls_num = static_cast<int>(result.label_ids.size());
    int num = cls_num > ret->size ? ret->size : cls_num;
    ret->size = num;
    for (size_t i = 0; i < num; i++) {
        ret->ids[i] = result.label_ids[i];
        ret->scores[i] = result.scores[i];
    }
    return true;
}

void free_model(void *model_handle, const ModeType &model_type) {

    if (model_type == ModeType::REC_MODEL) {
        delete static_cast<fastdeploy::vision::classification::MobileNetv3 *>(model_handle);
    } else if (model_type == ModeType::DET_MODEL) {
        delete static_cast<fastdeploy::vision::detection::PPYOLOE *>(model_handle);
    } else {
        std::cerr << "model type only supported in [ModeType::REC_MODEL,ModeType::DET_MODEL ]" << std::endl;
    }
}

int32_t get_local_index_by_rect(const Box &box, int w, int h) {
    auto mid_w = static_cast<float >(w) / 2.0f;
    auto mid_h = static_cast<float >(h) / 2.0f;
    Box box_0{0, 0, mid_w, mid_h};
    Box box_1{0, mid_h, mid_w, static_cast<float >(h)};
    Box box_2{mid_w, 0, static_cast<float >(w), mid_h};
    Box box_3{mid_w, mid_h, static_cast<float >(w), static_cast<float >(h)};
    if (is_contain_by_rect(box, box_0)) return 1;
    if (is_contain_by_rect(box, box_1)) return 3;
    if (is_contain_by_rect(box, box_2)) return 0;
    if (is_contain_by_rect(box, box_3)) return 2;
    return -1;
}

int32_t get_local_index_by_center_point(const Box &box, int w, int h) {
    auto mid_w = static_cast<float >(w) / 2.0f;
    auto mid_h = static_cast<float >(h) / 2.0f;
    Box box_0{0, 0, mid_w, mid_h};
    Box box_1{0, mid_h, mid_w, static_cast<float >(h)};
    Box box_2{mid_w, 0, static_cast<float >(w), mid_h};
    Box box_3{mid_w, mid_h, static_cast<float >(w), static_cast<float >(h)};
    float cent_point_x = (box.x_min + box.x_max) / 2;
    float cent_point_y = (box.y_min + box.y_max) / 2;
    if (is_contain_by_point(cent_point_x, cent_point_y, box_0)) return 1;
    if (is_contain_by_point(cent_point_x, cent_point_y, box_1)) return 3;
    if (is_contain_by_point(cent_point_x, cent_point_y, box_2)) return 0;
    if (is_contain_by_point(cent_point_x, cent_point_y, box_3)) return 2;
    return -1;
}


bool is_contain_by_rect(const Box &inner_box, const Box &outer_box) {
    if (inner_box.x_min > outer_box.x_min &&
        inner_box.x_max < outer_box.x_max &&
        inner_box.y_min > outer_box.y_min &&
        inner_box.y_min < outer_box.y_max) {
        return true;
    }
    return false;
}

bool is_contain_by_point(float center_x, float center_y, const Box &outer_box) {
    if (center_x > outer_box.x_min &&
        center_x < outer_box.x_max &&
        center_y > outer_box.y_min &&
        center_y < outer_box.y_max) {
        return true;
    }
    return false;
}

void free_ret_result(char *ret) {
    free(ret);
}

void obj_detection_str(model_handle_t model_handle, const char *image_path, float vis_threshold, char *ret) {
    Json::Value root;
    DetResult result;
    cv::Mat img = cv::imread(image_path);
    void *out_buffer = malloc(img.total() * img.elemSize());
    //{0:"square", 1:"triangle",2:"circle"}
    obj_detection(model_handle, img.data, out_buffer, img.cols, img.rows, &result, vis_threshold, true);
    Json::Value dst_root, sub, box_sub;
    for (size_t i = 0; i < result.size; i++) {
        box_sub["x_min"] = result.box[i].x_min;
        box_sub["x_max"] = result.box[i].x_max;
        box_sub["y_min"] = result.box[i].y_min;
        box_sub["y_max"] = result.box[i].y_max;
        sub["box"] = box_sub;
        sub["score"] = result.scores[i];
        sub["label"] = result.label_ids[i];
        sub["location"] = result.local_index[i];
        dst_root.append(sub);
    }
    std::cout << "[info] result is:" << std::endl;
    std::cout << dst_root << std::endl;
    Json::StreamWriterBuilder writer_builder;
    const std::string json_file = Json::writeString(writer_builder, dst_root);
//    ret = (char *) malloc(json_file.length());
    memcpy(ret, json_file.c_str(), json_file.length());
    free(out_buffer);
}
