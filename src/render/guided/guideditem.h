#pragma once
#include "common.h"
#include "raytracing.h"
#include "render/shared.h"

#include "render/wavefront/workitem.h"
#include "render/guided/parameters.h"

KRR_NAMESPACE_BEGIN

using namespace shader;



struct BsdfEvalWorkItem {
	uint itemId;		
};

struct GuidedInferenceWorkItem {
	uint itemId; 
};

struct RadianceRecordItem {
	Color L;			
	Color thp;			
	Vector3f pos;		
	Vector2f dir;		
	float wiPdf;		
	float bsdfPdf;		
	float misWeight;	
	bool delta;			
	
	Color bsdfVal;		
	Vector2f wo;		
	Vector2f normal;	
	float roughness;	

	KRR_CALLABLE void record(const Color& r) { L += r; }
};

struct GuidedPixelState {
	RadianceRecordItem records[MAX_TRAIN_DEPTH];
	uint curDepth{};
};

struct GuidedInput {
	Vector3f pos;	
#if GUIDED_PRODUCT_SAMPLING
	Vector3f dir;	
#endif
#if GUIDED_AUXILIARY_INPUT
	float auxiliary[N_DIM_AUXILIARY_INPUT];
#endif
};


struct GuidedOutput {
	Vector2f dir;
	Color L;
	float wiPdf;
#if GUIDED_LEARN_SELECTION
	float bsdfPdf;
#endif
};

#pragma warning (push, 0)
#pragma warning (disable: ALL_CODE_ANALYSIS_WARNINGS)
#include "render/guided/guideditem_soa.h"
#pragma warning (pop)

KRR_NAMESPACE_END