#pragma once
#include <atomic>

#include "common.h"
#include "logger.h"
#include "util/math_utils.h"
#include "device/cuda.h"

#include "render/profiler/profiler.h"
#include "render/wavefront/workqueue.h"
#include "render/guided/guideditem.h"

KRR_NAMESPACE_BEGIN

struct GuidedPixelStateBuffer : public SOA<GuidedPixelState> {
public:
	GuidedPixelStateBuffer() = default;
	GuidedPixelStateBuffer(int n, Allocator alloc) : SOA<GuidedPixelState>(n, alloc) {}

	KRR_CALLABLE void reset(int pixelId) {
		// reset a guided state (call when begining a new frame...
		curDepth[pixelId] = 0;
	}

	/* Records raw (unnormalized) vertex data along the path of the current pixel */
	KRR_CALLABLE void incrementDepth(int pixelId,
		const Ray& ray,				// current scattered ray
		const Color& thp,			// current throughput
		const float pdf,			// effective pdf of the scatter direction
		const float bsdfPdf = 0,	// bsdf pdf for the scatter direction
		const float misWeight = 0,	// effectiveGuidedPdf / combinedWiPdf
		const Color& L = {},		// current radiance 
		bool delta = false,			// is this scatter event sampled from a delta lobe?
		/* below optional / auxiliary data */
		const Color& bsdfVal = {},	// bsdf value for modeling 5-D product distribution
		const ShadingData& sd = {}	// may obtain other auxiliary data from this
	) {
		int depth = curDepth[pixelId];
		if (depth >= MAX_TRAIN_DEPTH) return;
		records[depth].L[pixelId]		  = L;
		records[depth].thp[pixelId]		  = thp;
		records[depth].pos[pixelId]		  = ray.origin;
		records[depth].dir[pixelId]		  = utils::cartesianToSpherical(ray.dir);
		records[depth].wiPdf[pixelId]	  = pdf;
		records[depth].bsdfPdf[pixelId]   = bsdfPdf; 
		records[depth].misWeight[pixelId] = misWeight;
		records[depth].delta[pixelId]	  = delta;
		curDepth[pixelId]				  = depth + 1;
#if GUIDED_AUXILIARY_INPUT
		records[depth].normal[pixelId]	  = utils::cartesianToSpherical(sd.frame.N);
		records[depth].roughness[pixelId] = sd.roughness;
#endif	
#if GUIDED_PRODUCT_SAMPLING
		records[depth].bsdfVal[pixelId] = bsdfVal;
		records[depth].wo[pixelId] = cartesianToSpherical(sd.wo);
#endif
	}

	/* Two types of radiance contribution call this routine: 
		Emissive intersection and Next event estimation. */
	KRR_CALLABLE void recordRadiance(int pixelId,
		const Color& L) {
		int depth = min(curDepth[pixelId], (uint)MAX_TRAIN_DEPTH);
		for (int i = 0; i < depth; i++) {
			// local radiance should be obtained via L / thp.
			const Color& prev = records[i].L[pixelId];
			records[i].L[pixelId] = prev + L;
		}
	}
};

class BsdfEvalQueue : public WorkQueue<BsdfEvalWorkItem> {
public:
	using WorkQueue::WorkQueue;
	using WorkQueue::push;

	KRR_CALLABLE int push(uint index) {
		return push(BsdfEvalWorkItem{ index });
	}
};

class GuidedInferenceQueue : public WorkQueue<GuidedInferenceWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	KRR_CALLABLE int push(uint index) { return push(GuidedInferenceWorkItem{ index }); }
};


KRR_NAMESPACE_END