#pragma once
#include "common.h"

#define GUIDED_AUXILIARY_INPUT		1
#define GUIDED_PRODUCT_SAMPLING		1
#define GUIDED_LEARN_SELECTION		0

constexpr unsigned int MAX_TRAIN_DEPTH		   = 8;
constexpr unsigned int NETWORK_ALIGNMENT	   = 16;
constexpr unsigned int NUM_VMF_COMPONENTS	   = 8 - GUIDED_LEARN_SELECTION;
constexpr unsigned int N_DIM_SPATIAL_INPUT	   = 3;
constexpr unsigned int N_DIM_DIRECTIONAL_INPUT = GUIDED_PRODUCT_SAMPLING ? 3 : 0;
constexpr unsigned int N_DIM_AUXILIARY_INPUT   = GUIDED_AUXILIARY_INPUT ? 3 : 0;
constexpr unsigned int N_DIM_INPUT  = N_DIM_DIRECTIONAL_INPUT + N_DIM_SPATIAL_INPUT + N_DIM_AUXILIARY_INPUT;
constexpr unsigned int N_DIM_VMF	= 4;
constexpr unsigned int N_DIM_OUTPUT = NUM_VMF_COMPONENTS * N_DIM_VMF + GUIDED_LEARN_SELECTION;
constexpr unsigned int N_DIM_PADDED_OUTPUT = 32;
constexpr unsigned int MAX_RESOLUTION	   = 1280 * 720;
constexpr int TRAIN_BUFFER_SIZE			   = MAX_TRAIN_DEPTH * MAX_RESOLUTION;
constexpr size_t TRAIN_BATCH_SIZE		   = 65'536 * 8;
constexpr size_t MIN_TRAIN_BATCH_SIZE	   = 65'536;
constexpr int MAX_INFERENCE_NUM			   = MAX_RESOLUTION;

constexpr float TRAIN_LOSS_SCALE = 128;
constexpr unsigned int LOSS_GRAPH_SIZE = 256;