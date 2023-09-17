//
// Created by DELL on 2023/6/6.
//

#pragma once

#include <iostream>
#include "fastdeploy/vision.h"

#ifndef CAPI
#define CAPI
#endif

#if defined(_WIN32)
#ifdef CAPI
#define API_EXPORT __declspec(dllexport)
#else
#define API_EXPORT __declspec(dllimport)
#endif  // CAPI
#else
#define API_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

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
    REC_MODEL = 0,
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
/// \param model_handle
/// \param model_dir
/// \param model_type
/// \param thread_num
/// \param use_gpu
/// \return
API_EXPORT bool init_model(model_handle_t *model_handle, const char *model_dir,
                           ModeType model_type, int thread_num, bool use_gpu);
///
/// \param model_handle
/// \param buffer
/// \param out_buffer
/// \param w
/// \param h
/// \param ret
/// \param vis_threshold
/// \param draw_text
/// \return
API_EXPORT bool obj_detection(model_handle_t model_handle, void *buffer,
                              void *out_buffer,
                              int w, int h,
                              DetResult *ret,
                              float vis_threshold = 0.5,
                              bool draw_text = false);
///
/// \param model_handle
/// \param buffer
/// \param w
/// \param h
/// \param ret
/// \return
API_EXPORT bool shape_classify(model_handle_t model_handle, void *buffer, int w, int h, ClsResult *ret);
///
/// \param model_handle
/// \param mode_type
API_EXPORT void free_model(model_handle_t model_handle, const ModeType &mode_type);

#ifdef __cplusplus
}
#endif