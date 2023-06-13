//
// Created by DELL on 2023/6/6.
//

#pragma once

#include <iostream>
#include "fastdeploy/vision.h"

#ifndef EXPORT_CAPI
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define API_EXPORT __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(__APPLE__))
#   define API_EXPORT __attribute__ ((visibility ("default")))
# endif
#endif

#define DET_NUM 200
#define CLS_NUM 10

#define FONT_FACE cv::FONT_HERSHEY_SIMPLEX || cv::FONT_HERSHEY_SIMPLEX
#define FONT_SCALE 0.4f
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
    int size;
} DetResult;

typedef struct ClsResult {
    int ids[CLS_NUM];
    float scores[CLS_NUM];
    int size;
} ClsResult;

bool init_det_model(void **model, const char *model_dir, const fastdeploy::RuntimeOption &opt);

bool init_rec_model(void **model, const char *model_dir, const fastdeploy::RuntimeOption &opt);


API_EXPORT bool init_model(void **model, const char *model_dir, ModeType model_type, int thread_num, bool use_gpu);


API_EXPORT bool obj_detection(void *model, void *buffer,
                                void *out_buffer,
                                int w, int h,
                                DetResult *ret,
                                float vis_threshold = 0.5,
                                bool draw_text = false);
API_EXPORT bool shape_classify(void *model, void *buffer, int w, int h, ClsResult *ret);

#ifdef __cplusplus
}
#endif
