/*This file should only be included in CUDA cpp files*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "device/atomic.h"
#include "common.h"
#include "guideditem.h"
#include "workqueue.h"
#include "interop.h"

#include "tiny-cuda-nn/common.h"

using precision_t = tcnn::network_precision_t;

KRR_NAMESPACE_BEGIN

enum class ELoss {
	L1,
	L2,
	LogL1,
	RelL2,
	NumTypes
};

enum class EDivergence {
	KL,
	ChiSquare,			
	NumTypes
};

enum class EOutputActivation {
	None = 0,
	ReLU,
	Logistic,
	Exponential,
	Softplus,
	NumTypes
};

enum class EOutputSlots {
	Lambda = 0,
	Kappa = 1,
	Theta = 2,
	Phi = 3,
	OutputDim
};

#define ACTIVATION_LAMBDA EOutputActivation::Exponential
#define ACTIVATION_KAPPA EOutputActivation::Exponential
#define ACTIVATION_THETA EOutputActivation::Logistic
#define ACTIVATION_PHI EOutputActivation::Logistic
#define ACTIVATION_PROB EOutputActivation::Logistic

#define LR_LAMBDA 1.f
#define LR_KAPPA 1.f
#define LR_THETA 1.f
#define LR_PHI 1.f
#define LR_PROB .01f

template <typename T>
class DeviceBuffer {
public:
	DeviceBuffer() = default;

	KRR_HOST DeviceBuffer(int n):
		mMaxSize(n) {
		cudaMalloc(&mData, n * sizeof(T));
	}

	KRR_CALLABLE int push(const T& w) {
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		(*this)[index % mMaxSize] = w;
		return index;
	}

	KRR_CALLABLE void clear() {
		mSize.store(0);
	}

	KRR_CALLABLE int size() const {
		return mSize.load();
	}

	KRR_CALLABLE T* data() { return mData; }

	KRR_CALLABLE T& operator [] (int index) {
		DCHECK_LT(index, mMaxSize);
		return mData[index];
	}

	KRR_CALLABLE DeviceBuffer& operator=(const DeviceBuffer& w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

protected:
	KRR_CALLABLE int allocateEntry() {
		return mSize.fetch_add(1);
	}

private:
	atomic<int> mSize;
	T* mData;
	int mMaxSize{ 0 };
};

class TrainBuffer {
public:
	TrainBuffer() = default;

	KRR_HOST TrainBuffer(int n) :
		mMaxSize(n) {
		cudaMalloc(&mInputs, n * sizeof(GuidedInput));
		cudaMalloc(&mOutputs, n * sizeof(GuidedOutput));
	}

	KRR_CALLABLE int push(const GuidedInput& input, 
		const GuidedOutput& output) {
		int index = allocateEntry();
		DCHECK_LT(index, mMaxSize);
		mInputs[index] = input;
		mOutputs[index] = output;
		return index;
	}

	KRR_CALLABLE void clear() { 
		mSize.store(0);		
	}

	KRR_CALLABLE int size() const {
#ifndef KRR_DEVICE_CODE
		CUDA_SYNC_CHECK();
		cudaDeviceSynchronize();
#endif
		return mSize.load();
	}

	KRR_CALLABLE void resize(int n) {
		if (mMaxSize) {
			cudaFree(mInputs);
			cudaFree(mOutputs);
		}
		cudaMalloc(&mInputs, n * sizeof(GuidedInput));
		cudaMalloc(&mOutputs, n * sizeof(GuidedOutput));
	}

	KRR_CALLABLE GuidedInput* inputs() const { return mInputs; }

	KRR_CALLABLE GuidedOutput* outputs() const { return mOutputs; }

	KRR_CALLABLE TrainBuffer& operator=(const TrainBuffer& w) {
		mSize.store(w.mSize);
		mMaxSize = w.mMaxSize;
		return *this;
	}

private:
	KRR_CALLABLE int allocateEntry() {
		return mSize.fetch_add(1);
	}

	atomic<int> mSize;
	GuidedInput* mInputs;
	GuidedOutput* mOutputs;
	int mMaxSize{ 0 };
};

KRR_CALLABLE Vector3f normalizeSpatialCoord(const Vector3f& coord, AABB aabb) {
	constexpr float inflation = 0.005f;
	aabb.inflate(aabb.diagonal().norm() * inflation);
	return Vector3f{ 0.5 } + (coord - aabb.center()) / aabb.diagonal();
}

__global__ void generate_training_data(const size_t nElements, Vector2ui trainPixelOffset,
									   Vector2ui trainPixelStride, Vector2ui trainingResolution,
									   Vector2ui resolution, TrainBuffer *trainBuffer,
									   GuidedPixelStateBuffer *guidedState, const AABB sceneAABB);

__global__ void compute_dL_doutput_divergence(const size_t nElements,
	precision_t* outputPrediction, GuidedOutput* outputReference,
	precision_t* dL_doutput, float* loss,
	float loss_scale, EDivergence divergence_type);

#if GUIDED_LEARN_SELECTION
__global__ void compute_dL_doutput_with_selection(const size_t nElements,
	precision_t* outputPrediction, GuidedOutput* outputReference,
	precision_t* dL_doutput, float* loss,
	float loss_scale, EDivergence divergence_type);
#endif

__global__ void generate_inference_data(const size_t nElements,
	ScatterRayQueue* scatterRayQueue, float* data, const AABB sceneAABB);

KRR_CALLABLE float network_to_params(float val, EOutputActivation activation) {
	static constexpr float exp_clamp_min = -10, exp_clamp_max = 15;
	switch (activation) {
	case EOutputActivation::None: return val;
	case EOutputActivation::ReLU: return val > 0.0f ? val : 0.0f;
	case EOutputActivation::Logistic: return logistic(val);
	case EOutputActivation::Exponential: return expf(clamp(val, exp_clamp_min, exp_clamp_max));
	case EOutputActivation::Softplus: return logf((1 + expf(val)));
	default: assert(false);
	}
	return 0.0f;
}

KRR_CALLABLE float d_network_to_d_params(float val, EOutputActivation activation) {
	static constexpr float exp_clamp_min = -10, exp_clamp_max = 15;
	switch (activation) {
	case EOutputActivation::None: return 1.0f;
	case EOutputActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
	case EOutputActivation::Logistic: { float fval = logistic(val); return fval * (1 - fval); };
	case EOutputActivation::Exponential: return expf(clamp(val, exp_clamp_min, exp_clamp_max));
	case EOutputActivation::Softplus: { float fval = expf(val); return fval / (fval + 1); };
	default: assert(false);
	}
	return 0.0f;
}

KRR_CALLABLE float loss_and_derivatives(ELoss type, float pred, float target, float* derivative = nullptr) {
	switch (type) {
	case ELoss::L1: {
		float diff = pred - target;
		if (derivative) *derivative = copysignf(1, diff);
		return fabs(diff);
	}
	case ELoss::L2: {
		float diff = pred - target;
		if (derivative) *derivative = 2 * diff;
		return pow2(diff);
	}
	case ELoss::LogL1: {
		float diff = pred - target;
		float divisor = fabs(diff) + 1;
		if (derivative) *derivative = copysignf(1/ divisor, diff);
		return log(divisor);
	}
	case ELoss::RelL2: {
		float diff = pred - target;
		float factor = 1 / (pow2(pred) + M_EPSILON);
		if (derivative) *derivative = 2 * diff * factor;
		return pow2(diff) * factor;
	}
	default: assert(false);
	}
	return 0;
}

KRR_NAMESPACE_END