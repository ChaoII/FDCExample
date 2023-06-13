//
// Created by DELL on 2023/6/6.
//

#include "defect_detect.h"

bool init_det_model(void **model, const char *model_dir, const fastdeploy::RuntimeOption &opt) {

    std::string model_path = std::string(model_dir) + "model.pdmodel";
    std::string param_path = std::string(model_dir) + "model.pdiparams";
    std::string config_path = std::string(model_dir) + "infer_cfg.yml";
    *model = new fastdeploy::vision::detection::PPYOLOE(model_path,
                                                        param_path,
                                                        config_path,
                                                        opt);
    if (!static_cast<fastdeploy::vision::detection::PPYOLOE *>(*model)->Initialized()) {
        return false;
    }
    return true;
}

bool init_rec_model(void **model, const char *model_dir, const fastdeploy::RuntimeOption &opt) {
    std::string model_path = std::string(model_dir) + "model.pdmodel";
    std::string param_path = std::string(model_dir) + "model.pdiparams";
    std::string config_path = std::string(model_dir) + "infer_cfg.yml";
    *model = new fastdeploy::vision::classification::MobileNetv3(model_path,
                                                                 param_path,
                                                                 config_path,
                                                                 opt);
    if (!static_cast<fastdeploy::vision::classification::MobileNetv3 *>(*model)->Initialized()) {
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

bool init_model(void **model, const char *model_dir, ModeType model_type, int thread_num, bool use_gpu) {
    fastdeploy::RuntimeOption opt;
    opt.UseOrtBackend();
    opt.SetCpuThreadNum(thread_num);
    if (use_gpu) {
        opt.UseGpu();
    }
    if (model_type == ModeType::REC_MODEL) {
        return init_rec_model(model, model_dir, opt);
    } else if (model_type == ModeType::DET_MODEL) {
        return init_det_model(model, model_dir, opt);
    } else {
        return false;
    }
}

bool obj_detection(void *model, void *buffer,
                     void *out_buffer,
                     int w, int h,
                     DetResult *ret,
                     float vis_threshold,
                     bool draw_text) {
    fastdeploy::vision::DetectionResult result;
    cv::Mat img = cv::Mat(h, w, CV_8UC3, buffer);
    static_cast<fastdeploy::vision::detection::PPYOLOE *>(model)->Predict(img, &result);
    int boxes_num = static_cast<int>(result.boxes.size());
    int num = boxes_num > DET_NUM ? DET_NUM : boxes_num;
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
        if (ret->scores[i] > vis_threshold) {
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
    memcpy(out_buffer, img.data, img.total() * img.elemSize());
    return true;
}


bool shape_classify(void *model, void *buffer, int w, int h, ClsResult *ret) {
    fastdeploy::vision::ClassifyResult result;
    cv::Mat img = cv::Mat(h, w, CV_8UC3, buffer);
    static_cast<fastdeploy::vision::classification::MobileNetv3 *>(model)->Predict(img, &result);
    int cls_num = static_cast<int>(result.label_ids.size());
    int num = cls_num > ret->size ? ret->size : cls_num;
    ret->size = num;
    for (size_t i = 0; i < num; i++) {
        ret->ids[i] = result.label_ids[i];
        ret->scores[i] = result.scores[i];
    }
    return true;
}