#pragma once

#include "window.h"
#include "sampler.h"
#include "scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"
#include "device/optix.h"
#include "render/wavefront/workqueue.h"
#include "render/guided/workqueue.h"

KRR_NAMESPACE_BEGIN

class PixelStateBuffer;
class GuidedPixelStateBuffer;

using namespace shader;

class TrainState {
public:
	KRR_CALLABLE bool isEnableGuiding() const { return enableGuiding; }

	KRR_CALLABLE bool isEnableTraining() const { return enableTraining; }

	KRR_CALLABLE bool isTrainingPixel(uint pixelId) const {
		Vector2ui pixel = { pixelId % resolution.x(), pixelId / resolution.x() };
		return enableTraining && (pixel.x() - trainPixelOffset.x()) % trainPixelStride.x() == 0 &&
			   (pixel.y() - trainPixelOffset.y()) % trainPixelStride.y() == 0;
	}

	KRR_CALLABLE Vector2ui getTrainingResolution() const {
		return (resolution + trainPixelStride - Vector2ui::Ones()) / trainPixelStride;
	}

	KRR_HOST void renderUI() {
		ImGui::Checkbox("Enable Guiding", &enableGuiding);
		ImGui::Checkbox("Enable Training", &enableTraining);
		ImGui::InputInt("Train Pixel Stride", (int*)&trainPixelStride);
	}

	bool enableTraining{ false };
	bool enableGuiding{ false };
	Vector2ui resolution;
	Vector2ui trainPixelOffset{ 0 };
	Vector2ui trainPixelStride{ 1 };
};

typedef struct {
	RayQueue* currentRayQueue;
	RayQueue* nextRayQueue;
	ShadowRayQueue* shadowRayQueue;
	MissRayQueue* missRayQueue;
	HitLightRayQueue* hitLightRayQueue;
	ScatterRayQueue* scatterRayQueue;
	
	PixelStateBuffer* pixelState;
	GuidedPixelStateBuffer* guidedState;
	TrainState trainState;
	rt::SceneData sceneData;
	OptixTraversableHandle traversable;
} LaunchParamsGuided;

KRR_NAMESPACE_END