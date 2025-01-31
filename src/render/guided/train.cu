#include "train.h"
#include "distribution.h"

KRR_NAMESPACE_BEGIN

KRR_CALLABLE Vector3f warp_direction_for_sh(const Vector3f& dir) {
	return (dir + Vector3f::Ones()) * 0.5f;
}

KRR_CALLABLE float warp_roughness_for_ob(const float roughness) {
	return 1 - expf(-roughness);
}

__global__ void generate_training_data(const size_t nElements, Vector2ui trainPixelOffset,
									   Vector2ui trainPixelStride, 
	Vector2ui trainingResolution, Vector2ui resolution, 
	TrainBuffer* trainBuffer, GuidedPixelStateBuffer* guidedState, const AABB sceneAABB) {
	// this costs about 0.5ms
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) return;

	Vector2ui trainigIdInTile{ tid % trainingResolution.x(), tid / trainingResolution.x() };
	Vector2ui pixel = trainigIdInTile.cwiseProduct(trainPixelStride) + trainPixelOffset;
	if (pixel.x() >= resolution.x() || pixel.y() >= resolution.y())
		return;

	int pixelId = pixel.x() + pixel.y() * resolution.x();
	
	int depth = guidedState->curDepth[pixelId];
	for (int curDepth = 0; curDepth < depth; curDepth++) {
		GuidedInput input = {};
		GuidedOutput output = {};
		
		const RadianceRecordItem& record = guidedState->records[curDepth][pixelId];
		if (record.delta) continue;
		input.pos = normalizeSpatialCoord(record.pos, sceneAABB);
#if GUIDED_PRODUCT_SAMPLING
		input.dir = warp_direction_for_sh(sphericalToCartesian(record.wo[0], record.wo[1]));
#endif
#if GUIDED_AUXILIARY_INPUT > 0
		input.auxiliary[0] = record.normal[0] * M_INV_PI;
		input.auxiliary[1] = record.normal[1] * M_INV_2PI;
		input.auxiliary[2] = warp_roughness_for_ob(record.roughness);
#endif 
		Color L = Color::Zero();
		for (int ch = 0; ch < Color::dim; ch++) {
			if (record.thp[ch] > M_EPSILON)
				L[ch] = record.L[ch] / record.thp[ch];
		}
		L *= record.misWeight;
		output.L   = L;
		output.dir = record.dir;
#if GUIDED_PRODUCT_SAMPLING
		output.L *= record.bsdfVal;
#endif
#if GUIDED_LEARN_SELECTION
		output.bsdfPdf = record.bsdfPdf;
#endif
		output.wiPdf = record.wiPdf;
		if (!(input.pos.hasNaN() || output.dir.hasNaN() 
#if GUIDED_PRODUCT_SAMPLING
			|| input.dir.hasNaN()
#endif
			|| isnan(output.wiPdf) || output.wiPdf == 0 || L.hasNaN()))
			trainBuffer->push(input, output);
	}
}

__global__ void compute_dL_doutput_divergence(
	const size_t nElements			/* number of threads */,
	precision_t* outputPrediction	/*[input] 4 x N_MIXTURE */,
	GuidedOutput* outputReference	/*[input] 3(RGB), MC estimate */,
	precision_t* dL_doutput			/*[output] 4 x N_MIXTURE */,
	float* likelihood				/*[output] 1 */,
	float loss_scale				/*[input] scale the loss so that it wont be too smol*/,
	EDivergence divergence_type		/*[input] divergence type */) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements) return;

	loss_scale /= nElements;
	precision_t* data = outputPrediction + tid * N_DIM_PADDED_OUTPUT;
	precision_t* gradient_data = dL_doutput + tid * N_DIM_PADDED_OUTPUT;
	MixedSphericalGaussianDistribution<NUM_VMF_COMPONENTS> dist(data);

	Vector3f wi = utils::sphericalToCartesian(outputReference[tid].dir[0], outputReference[tid].dir[1]);

	if (wi.hasNaN()) printf("Found NaN in wi directions!\n");

	float Li		= outputReference[tid].L.mean();
	float wiPdf		= outputReference[tid].wiPdf + M_EPSILON;
	float guidePdf	= dist.gradients_probability(wi, gradient_data) + M_EPSILON;
	float prefix	= -Li / wiPdf / guidePdf * loss_scale;
	likelihood[tid] = -Li / wiPdf * logf(guidePdf);

	for (int sg = 0; sg < NUM_VMF_COMPONENTS; sg++) {
		precision_t* cur_params = data + sg * N_DIM_VMF;
		float lambda_r = cur_params[0], kappa_r = cur_params[1],
			theta_r = cur_params[2], phi_r = cur_params[3];
		precision_t* cur_gradient = gradient_data + sg * N_DIM_VMF;
		cur_gradient[0] = LR_LAMBDA * prefix * (float)cur_gradient[0] * d_network_to_d_params(lambda_r, ACTIVATION_LAMBDA);
		cur_gradient[1] = LR_KAPPA * prefix * (float)cur_gradient[1] * d_network_to_d_params(kappa_r, ACTIVATION_KAPPA);
		cur_gradient[2] = LR_THETA * prefix * (float)cur_gradient[2] * M_PI * d_network_to_d_params(theta_r, ACTIVATION_THETA);
		cur_gradient[3] = LR_PHI * prefix * (float)cur_gradient[3] * M_2PI * d_network_to_d_params(phi_r, ACTIVATION_PHI);
	}

#define GRADIENT_CHECK_NAN
#define GRADIENT_CHECK_LARGE

#ifdef GRADIENT_CHECK_NAN
	for (int i = 0; i < N_DIM_OUTPUT; i++) 
		if(isnan((float)gradient_data[i]) || isinf((float)gradient_data[i])) 
			gradient_data[i] = (precision_t)0.f;
#endif
	
#ifdef GRADIENT_CHECK_LARGE
	constexpr float GRADIENT_BACKPROP_CLAMP = 150.f;
	for (int i = 0; i < N_DIM_OUTPUT; i++) {
		if ((float) gradient_data[i] > GRADIENT_BACKPROP_CLAMP)
			printf("Possibly too large gradient %.2f at slot %d\n", 
				(float) gradient_data[i], i % N_DIM_VMF);
		gradient_data[i] = min(GRADIENT_BACKPROP_CLAMP, (float) gradient_data[i]);
	}
#endif
}

#if GUIDED_LEARN_SELECTION
__global__ void compute_dL_doutput_with_selection(
	const size_t nElements ,
	precision_t *outputPrediction,
	GuidedOutput *outputReference,
	precision_t *dL_doutput /*[output] 4 x N_MIXTURE */, float *likelihood /*[output] 1 */,
	float loss_scale /*[input] scale the loss so that it wont be too smol*/,
	EDivergence divergence_type /*[input] divergence type */) {
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= nElements)
		return;

	loss_scale /= nElements;
	precision_t *data		   = outputPrediction + tid * N_DIM_PADDED_OUTPUT;
	precision_t *gradient_data = dL_doutput + tid * N_DIM_PADDED_OUTPUT;
	MixedSGWithSelection<NUM_VMF_COMPONENTS> dist(data);
	Vector3f wi =
		utils::sphericalToCartesian(outputReference[tid].dir[0], outputReference[tid].dir[1]);

	if (wi.hasNaN()) printf("Found NaN in wi directions!\n");

	float Li	= outputReference[tid].L.mean();
	float bsdfPdf		= outputReference[tid].bsdfPdf;
	float selectionProb = dist.getSelectionProbability();
	float guidePdf		= dist.gradients_probability(wi, bsdfPdf, gradient_data) + M_EPSILON;
	float wiPdf			= selectionProb * guidePdf + (1 - selectionProb) * bsdfPdf;

	likelihood[tid] = -Li / wiPdf * logf(wiPdf);
	float prefix	= -Li / wiPdf / wiPdf * loss_scale;

	for (int sg = 0; sg < NUM_VMF_COMPONENTS; sg++) {
		precision_t *cur_params = data + sg * N_DIM_VMF;
		float lambda_r = cur_params[0], kappa_r = cur_params[1], theta_r = cur_params[2],
			  phi_r				  = cur_params[3];
		precision_t *cur_gradient = gradient_data + sg * N_DIM_VMF;
		cur_gradient[0] = LR_LAMBDA * prefix * (float) cur_gradient[0] * d_network_to_d_params(lambda_r, ACTIVATION_LAMBDA);
		cur_gradient[1] = LR_KAPPA * prefix * (float) cur_gradient[1] * d_network_to_d_params(kappa_r, ACTIVATION_KAPPA);
		cur_gradient[2] = LR_THETA * prefix * (float) cur_gradient[2] * M_PI * d_network_to_d_params(theta_r, ACTIVATION_THETA);
		cur_gradient[3] = LR_PHI * prefix * (float) cur_gradient[3] * M_2PI * d_network_to_d_params(phi_r, ACTIVATION_PHI);
	}
	const int idx_selection = N_DIM_VMF * NUM_VMF_COMPONENTS;
	gradient_data[idx_selection] = LR_PROB * prefix * (float) gradient_data[idx_selection] *
								   d_network_to_d_params(data[idx_selection], ACTIVATION_PROB);

	if (tid == 7272) printf("Current SEL prob = %f\n", selectionProb);
}
#endif

__global__ void generate_inference_data(const size_t nElements,
	ScatterRayQueue* scatterRayQueue, float* data, const AABB sceneAABB) {
	const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= scatterRayQueue->size()) return;
	uint data_idx		  = i * N_DIM_INPUT;
	const ShadingData &sd = scatterRayQueue->operator[](i).operator krr::ScatterRayWorkItem().sd;
	Vector3f pos		  = normalizeSpatialCoord(sd.pos, sceneAABB);
	*(Vector3f *) &data[data_idx] = pos;
#if GUIDED_PRODUCT_SAMPLING
	*(Vector3f *) &data[data_idx + N_DIM_SPATIAL_INPUT] = warp_direction_for_sh(sd.wo.normalized());
#endif
#if GUIDED_AUXILIARY_INPUT
	*(Vector2f *) &data[data_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT] =
		utils::cartesianToSphericalNormalized(sd.frame.N);
	data[data_idx + N_DIM_SPATIAL_INPUT + N_DIM_DIRECTIONAL_INPUT + 2] =
		warp_roughness_for_ob(sd.roughness);
#endif
}

KRR_NAMESPACE_END