//
// Created by DELL on 2023/6/6.
//

#pragma once

#include <iostream>
#include "fastdeploy/vision.h"
#include "exports.h"

#define DET_NUM 200
#define CLS_NUM 10

#define FONT_FACE cv::FONT_HERSHEY_SIMPLEX || cv::FONT_HERSHEY_SIMPLEX
#define FONT_SCALE 0.4f
typedef void *model_handle_t;

typedef struct SortedArray {
    int index;
    float value;
} SortedArray;

std::vector<SortedArray> sort_det_result(fastdeploy::vision::DetectionResult &result);

#ifdef __cplusplus
extern "C" {
#endif

enum ModeType {
    // 识别模型
    REC_MODEL = 0,
    // 检测模型
    DET_MODEL,
};


typedef struct Box {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
} Box;

typedef struct DetResult {
    Box box[DET_NUM];
    float scores[DET_NUM];
    int label_ids[DET_NUM];
    int32_t local_index[DET_NUM];
    int size;
} DetResult;

typedef struct ClsResult {
    int ids[CLS_NUM];
    float scores[CLS_NUM];
    int size;
} ClsResult;

int32_t get_local_index(const Box &box, int w, int h);

bool is_contain(const Box &inner_box, const Box &outer_box);

bool init_det_model(model_handle_t *model_handle, const char *model_dir, const fastdeploy::RuntimeOption &opt);

bool init_rec_model(model_handle_t *model_handle, const char *model_dir, const fastdeploy::RuntimeOption &opt);

///
/// \param model_handle 模型句柄
/// \param model_dir 模型目录
/// \param model_type 模型类型0分类模型，1检测模型
/// \param thread_num 线程数
/// \param use_gpu 是否开启GPU
/// \return
FDD_EXPORT bool init_model(model_handle_t *model_handle, const char *model_dir,
                           ModeType model_type, int thread_num, bool use_gpu);
///
/// \param model_handle 模型句柄
/// \param buffer 输入图像raw buffer
/// \param out_buffer 输出图像raw buffer
/// \param w 图像宽
/// \param h 图像高
/// \param ret 模型推理结果结构体
/// \param vis_threshold 可视化阈值
/// \param draw_text 是否图中绘制结果
/// \return
FDD_EXPORT bool obj_detection(model_handle_t model_handle, void *buffer,
                              void *out_buffer,
                              int w, int h,
                              DetResult *ret,
                              float vis_threshold = 0.5,
                              bool draw_text = false);
///
/// \param model_handle 模型句柄
/// \param buffer 输入图像raw buffer
/// \param w 图像宽
/// \param h 图像高
/// \param ret 分类模型结果结构体
/// \return
FDD_EXPORT bool shape_classify(model_handle_t model_handle, void *buffer, int w, int h, ClsResult *ret);
///
/// \param model_handle 模型句柄
/// \param mode_type 模型类型
FDD_EXPORT void free_model(model_handle_t model_handle, const ModeType &mode_type);

#ifdef __cplusplus
}
#endif