#pragma once
#include "json.hpp"
#include "tiny-cuda-nn/common.h"

#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"
#include "renderpass.h"

#include "device/buffer.h"
#include "device/context.h"
#include "device/cuda.h"
#include "device/optix.h"
#include "train.h"
#include "guided.h"
#include "workqueue.h"
#include "util/task.h"

namespace tcnn {
	template <typename T> class Loss;
	template <typename T> class Optimizer;
	template <typename T> class Encoding;
	template <typename T> class GPUMemory;
	template <typename T> class GPUMatrixDynamic;
	template <typename T, typename PARAMS_T> class Network;
	template <typename T, typename PARAMS_T, typename COMPUTE_T> class Trainer;
	template <uint32_t N_DIMS, uint32_t RANK, typename T> class TrainableBuffer;
}

KRR_NAMESPACE_BEGIN

using nlohmann::json;
using precision_t = tcnn::network_precision_t;

class Film;

class GuidedPathTracer : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<GuidedPathTracer>;
	KRR_REGISTER_PASS_DEC(GuidedPathTracer);
	enum class RenderMode {Interactive, Offline};

	GuidedPathTracer() = default;
	GuidedPathTracer(Scene& scene);
	~GuidedPathTracer() = default;

	void resize(const Vector2i& size) override;
	void setScene(Scene::SharedPtr scene) override;
	void beginFrame(RenderContext *context) override;
	void endFrame(RenderContext *context) override;
	void render(RenderContext* context) override;
	void renderUI() override;
	void finalize() override;

	void initialize();

	string getName() const override { return "GuidedPathTracer"; }

	void handleIntersections();
	void handleEmissiveHit();
	void handleMiss();
	void handleBsdfSampling();
	void hendleGuidedSampling();
	void generateCameraRays(int sampleId);
	void traceClosest(int depth);
	void traceShadow();

	KRR_CALLABLE RayQueue* currentRayQueue(int depth) { return rayQueue[depth & 1]; }
	KRR_CALLABLE RayQueue* nextRayQueue(int depth) { return rayQueue[(depth & 1) ^ 1]; }

	template <typename... Args>
	KRR_DEVICE_FUNCTION void debugPrint(uint pixelId, const char *fmt, Args &&...args);

	
	void trainStep();
	void resetTraining();
	void inferenceStep();

	OptixBackend* backend;
	Camera::CameraData* camera{ };
	LightSampler lightSampler;

	
	RayQueue* rayQueue[2]{ };				
	MissRayQueue* missRayQueue{ };
	HitLightRayQueue* hitLightRayQueue{ };
	ShadowRayQueue* shadowRayQueue{ };	
	ScatterRayQueue* scatterRayQueue{ };	
	BsdfEvalQueue* bsdfEvalQueue{ };
	GuidedInferenceQueue *guidedRayQueue{};

	PixelStateBuffer* pixelState;
	GuidedPixelStateBuffer* guidedState;
	TrainBuffer* trainBuffer;
	
	bool debugOutput{ };
	uint debugPixel{ };
	int maxQueueSize;
	int frameId{ 0 };

	
	int samplesPerPixel{ 1 };
	int maxDepth{ 10 };
	float probRR{ 1 };
	bool enableNEE{ true };
	bool enableClamp{ false };
	float clampMax{ 1e4f };

	void resetNetwork(json config);
	bool oneStep{}, trainDebug{};

	
	RenderMode renderMode{ RenderMode::Interactive };
	RenderTask task{};
	float trainingBudgetTime{ 0.3 };
	float trainingBudgetSpp{ 0.25 };
	bool isTrainingFinished{ false };
	bool autoTrain{ false };
	bool discardTraining{ false };		
	Film *renderedImage{ nullptr };

	
	class Guidance {
	public:
		KRR_CALLABLE bool isEnableGuiding() const { return trainState.isEnableGuiding(); }
		
		KRR_CALLABLE bool isEnableTraining() const { return trainState.isEnableTraining(); }
		
		KRR_CALLABLE bool isEnableGuiding(uint depth) const {
			return trainState.isEnableGuiding() && depth < maxGuidedDepth;
		}

		KRR_CALLABLE bool isEnableTraining(uint depth) const { 
			return trainState.isEnableTraining() && depth < maxTrainDepth;
		}

		KRR_HOST void beginFrame() {
			trainState.trainPixelOffset = trainState.trainPixelStride <= 1 ?
				0 : sampler.get1D() * trainState.trainPixelStride;
		}

		KRR_HOST void renderUI();
		
		KRR_CALLABLE bool isTrainingPixel(uint pixelId) const {
			return trainState.isTrainingPixel(pixelId);
		}
		
		TrainState trainState;
		PCGSampler sampler;
		cudaStream_t stream{};
		bool cosineAware{ true };
		bool misAware{ false };
		bool sampleWeighting{ false };
		bool guideTransmissive{ true };
		bool productWithCosine{ false };
		float bsdfSamplingFraction{ 0.5 };
		uint maxGuidedDepth{ 10 };
		uint maxTrainDepth{ 3 };
		uint batchPerFrame{ 5 };
		uint batchSize{ TRAIN_BATCH_SIZE };

		json config;
		
		EDivergence divergence_type{ EDivergence::KL };
		std::shared_ptr<tcnn::Network<float, precision_t>> network;
		std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer;
		std::shared_ptr<tcnn::Encoding<precision_t>> encoding;
		std::shared_ptr<tcnn::Loss<precision_t>> loss;
		std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer;
	} mGuiding;

	friend void to_json(json &j, const GuidedPathTracer &p) {
		j = json{
			{ "mode", p.renderMode},
			{ "nee", p.enableNEE },
			{ "max_depth", p.maxDepth },
			{ "rr", p.probRR },			
			{ "enable_clamp", p.enableClamp },
			{ "clamp_max", p.clampMax },
			{ "bsdf_fraction", p.mGuiding.bsdfSamplingFraction },
			{ "cosine_aware",  p.mGuiding.cosineAware},
			{ "mis_aware", p.mGuiding.misAware },
			{ "guide_transmissive", p.mGuiding.guideTransmissive },
			{ "product_cosine", p.mGuiding.productWithCosine },
			{ "max_train_depth", p.mGuiding.maxTrainDepth },
			{ "max_guided_depth", p.mGuiding.maxGuidedDepth },
			{ "auto_train", p.autoTrain },
			{ "train_budget_spp", p.trainingBudgetSpp},
			{ "train_budget_time", p.trainingBudgetTime},
			{ "batch_per_frame", p.mGuiding.batchPerFrame },
			{ "batch_size", p.mGuiding.batchSize },
			{ "budget", p.task }
		};
	}

	friend void from_json(const json &j, GuidedPathTracer &p) {
		p.renderMode  = j.value("mode", GuidedPathTracer::RenderMode::Interactive);
		p.enableNEE	  = j.value("nee", true);
		p.maxDepth	  = j.value("max_depth", 6);
		p.probRR	  = j.value("rr", 0.8);
		p.enableClamp = j.value("enable_clamp", false);
		p.clampMax	  = j.value("clamp_max", 1e4f);
		p.mGuiding.bsdfSamplingFraction = j.value("bsdf_fraction", 0.5);
		p.mGuiding.sampleWeighting		= j.value("sample_weighting", false);
		p.mGuiding.cosineAware			= j.value("cosine_aware", true);
		p.mGuiding.misAware				= j.value("mis_aware", true);
		p.mGuiding.guideTransmissive	= j.value("guide_transmissive", true);
		p.mGuiding.productWithCosine	= j.value("product_cosine", false);
		p.mGuiding.maxTrainDepth		= j.value("max_train_depth", 3);
		p.mGuiding.maxGuidedDepth		= j.value("max_guided_depth", 10);
		p.autoTrain						= j.value("auto_train", false);
		p.discardTraining				= j.value("discard_training", false);
		p.trainingBudgetSpp				= j.value("training_budget_spp", 0.25f);
		p.trainingBudgetTime			= j.value("training_budget_time", 0.3f);
		p.mGuiding.batchPerFrame		= j.value("batch_per_frame", 5);
		p.mGuiding.batchSize			= j.value("batch_size", TRAIN_BATCH_SIZE);
		p.task							= j.value("budget", RenderTask{});

		CHECK_LOG(p.mGuiding.maxTrainDepth <= MAX_TRAIN_DEPTH, 
				"Max train depth %d exceeds limit %d!", p.mGuiding.maxTrainDepth,
				  MAX_TRAIN_DEPTH);

		if (j.contains("config")) {
			string config_path = j.at("config");
			std::ifstream f(config_path);
			if (f.fail())
				logFatal("Open network config file failed!");
			json config = json::parse(f, nullptr, true, true);
			p.resetNetwork(config["nn"]);
		} else {
			Log(Warning, "Network config do not specified!"
				"Assuming guiding is disabled.");
		}
			
	}
};

KRR_ENUM_DEFINE(GuidedPathTracer::RenderMode, {
	{GuidedPathTracer::RenderMode::Interactive, "interactive"},
	{GuidedPathTracer::RenderMode::Offline, "offline"},
})

KRR_NAMESPACE_END