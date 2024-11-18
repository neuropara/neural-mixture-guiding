#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include "train.h"
#include "distribution.h"
#include "integrator.h"
#include "render/profiler/profiler.h"
#include "render/guided/network.h"
#include "util/ema.h"
#include "util/film.h"

KRR_NAMESPACE_BEGIN
extern "C" char GUIDED_PTX[];
using namespace tcnn;

template <typename T>
using GPUMatrix = tcnn::GPUMatrix<T, tcnn::MatrixLayout::ColumnMajor>;

namespace {
	
	tcnn::GPUMemory<precision_t> trainOutputBuffer;
	tcnn::GPUMemory<float> inferenceOutputBuffer;
	tcnn::GPUMemory<precision_t> gradientBuffer;
	tcnn::GPUMemory<float> lossBuffer;
	tcnn::GPUMemory<float> inferenceInputBuffer;

	
	std::vector<float> lossGraph(LOSS_GRAPH_SIZE, 0);
	size_t numLossSamples{ 0 };
	size_t numTrainingSamples{ 0 };
	Ema curLossScalar{Ema::Type::Time, 50};
}

GuidedPathTracer::GuidedPathTracer(Scene& scene){
	initialize();
	setScene(std::shared_ptr<Scene>(&scene));
}

template <typename... Args> 
KRR_DEVICE_FUNCTION void GuidedPathTracer::debugPrint(uint pixelId, const char *fmt, Args &&...args) {
	if (pixelId == debugPixel)
		printf(fmt, std::forward<Args>(args)...);
}

void GuidedPathTracer::initialize(){
	Allocator& alloc = *gpContext->alloc;
	maxQueueSize	 = getFrameSize()[0] * getFrameSize()[1];
	CUDA_SYNC_CHECK();	
	for (int i = 0; i < 2; i++) {
		if (rayQueue[i]) rayQueue[i]->resize(maxQueueSize, alloc);
		else rayQueue[i] = alloc.new_object<RayQueue>(maxQueueSize, alloc);
	}
	if (missRayQueue)  missRayQueue->resize(maxQueueSize, alloc);
	else missRayQueue = alloc.new_object<MissRayQueue>(maxQueueSize, alloc);
	if (hitLightRayQueue)  hitLightRayQueue->resize(maxQueueSize, alloc);
	else hitLightRayQueue = alloc.new_object<HitLightRayQueue>(maxQueueSize, alloc);
	if (shadowRayQueue) shadowRayQueue->resize(maxQueueSize, alloc);
	else shadowRayQueue = alloc.new_object<ShadowRayQueue>(maxQueueSize, alloc);
	if (scatterRayQueue) scatterRayQueue->resize(maxQueueSize, alloc);
	else scatterRayQueue = alloc.new_object<ScatterRayQueue>(maxQueueSize, alloc);
	if (bsdfEvalQueue) bsdfEvalQueue->resize(maxQueueSize, alloc);
	else bsdfEvalQueue = alloc.new_object<BsdfEvalQueue>(maxQueueSize, alloc);
	if (guidedRayQueue) guidedRayQueue->resize(maxQueueSize, alloc);
	else guidedRayQueue = alloc.new_object<GuidedInferenceQueue>(maxQueueSize, alloc);

	if (pixelState) pixelState->resize(maxQueueSize, alloc);
	else pixelState = alloc.new_object<PixelStateBuffer>(maxQueueSize, alloc);
	if (guidedState) guidedState->resize(maxQueueSize, alloc);
	else guidedState = alloc.new_object<GuidedPixelStateBuffer>(maxQueueSize, alloc);
	if (!trainBuffer) trainBuffer = alloc.new_object<TrainBuffer>(TRAIN_BUFFER_SIZE);
	cudaDeviceSynchronize();	
	if (!camera) camera = alloc.new_object<Camera::CameraData>();
	
	if (renderedImage) renderedImage->resize(getFrameSize());
	else renderedImage = alloc.new_object<Film>(getFrameSize());
	renderedImage->reset();
	CUDA_SYNC_CHECK();
	task.reset();
}

void GuidedPathTracer::traceClosest(int depth) {
	PROFILE("Trace intersect rays");
	static LaunchParamsGuided params = {};
	params.traversable				 = backend->getRootTraversable();
	params.sceneData				 = backend->getSceneData();
	params.currentRayQueue			 = currentRayQueue(depth);
	params.missRayQueue				 = missRayQueue;
	params.hitLightRayQueue			 = hitLightRayQueue;
	params.scatterRayQueue			 = scatterRayQueue;
	params.nextRayQueue				 = nextRayQueue(depth);
	backend->launch(params, "Closest", maxQueueSize, 1, 1);
}

void GuidedPathTracer::traceShadow() {
	PROFILE("Trace shadow rays");
	static LaunchParamsGuided params = {};
	params.traversable				 = backend->getRootTraversable();
	params.sceneData				 = backend->getSceneData();
	params.shadowRayQueue			 = shadowRayQueue;
	params.pixelState				 = pixelState;
	params.guidedState				 = guidedState;
	params.trainState				 = mGuiding.trainState;
	backend->launch(params, "Shadow", maxQueueSize, 1, 1);
}

void GuidedPathTracer::handleEmissiveHit(){
	PROFILE("Process intersected rays");
	ForAllQueued(hitLightRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const HitLightWorkItem & w){
		Color Le = w.light.L(w.p, w.n, w.uv, w.wo);
		float misWeight = 1;
		if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
			Light light = w.light;
			Interaction intr(w.p, w.wo, w.n, w.uv);
			float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(light);
			misWeight = evalMIS(w.pdf, lightPdf);
		}
		Color contrib = Le * w.thp * misWeight;
		pixelState->addRadiance(w.pixelId, contrib);
		if (mGuiding.isTrainingPixel(w.pixelId)) 
			guidedState->recordRadiance(w.pixelId, contrib);
	});
}

void GuidedPathTracer::handleMiss(){
	PROFILE("Process escaped rays");
	const rt::SceneData &sceneData = mScene->mSceneRT->getSceneData();
	ForAllQueued(missRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MissRayWorkItem& w) {
		Color L = {};
		Interaction intr(w.ray.origin);
		for (const rt::InfiniteLight& light : sceneData.infiniteLights) {
			float misWeight = 1;
			if (enableNEE && w.depth && !(w.bsdfType & BSDF_SPECULAR)) {
				float lightPdf = light.pdfLi(intr, w.ctx) * lightSampler.pdf(&light);
				misWeight = evalMIS(w.pdf, lightPdf);
			}
			L += light.Li(w.ray.dir) * misWeight;
		}
		Color contrib = L * w.thp;
		pixelState->addRadiance(w.pixelId, contrib);
		if (mGuiding.isTrainingPixel(w.pixelId))
			guidedState->recordRadiance(w.pixelId, contrib);
	});
}

void GuidedPathTracer::handleIntersections(){
	PROFILE("Process intersections");
	float* guidingOutput = inferenceOutputBuffer.data();
	ForAllQueued(scatterRayQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(ScatterRayWorkItem & w) {
		Sampler sampler = &pixelState->sampler[w.pixelId];
		
		if (sampler.get1D() >= probRR) return;
		w.thp /= probRR;
		int tid = blockIdx.x * blockDim.x + threadIdx.x; 
		const ShadingData& sd = w.sd;
		
		const BSDFType bsdfType = sd.getBsdfType();
		float *dist_data		= guidingOutput + tid * N_DIM_OUTPUT;
		float bsdfSamplingFraction = mGuiding.bsdfSamplingFraction;
#if GUIDED_LEARN_SELECTION
		bsdfSamplingFraction = 1 - 
			MixedSGWithSelection<NUM_VMF_COMPONENTS>::extractSelectionProbability(dist_data);
#endif

	
		if (enableNEE && (bsdfType & BSDF_SMOOTH)) {
			SampledLight sampledLight = lightSampler.sample(sampler.get1D());
			Light light = sampledLight.light;
			LightSample ls = light.sampleLi(sampler.get2D(), LightSampleContext{ sd.pos, sd.frame.N });
			Ray shadowRay = sd.getInteraction().spawnRay(ls.intr);
			Vector3f woLocal = sd.frame.toLocal(sd.wo);
			Vector3f wiWorld = normalize(shadowRay.dir);
			Vector3f wiLocal = sd.frame.toLocal(wiWorld);

			float lightPdf = sampledLight.pdf * ls.pdf;
		
			float scatterPdf{ 0 };
			if (!mGuiding.isEnableGuiding(w.depth) || bsdfSamplingFraction > 0) {
				scatterPdf = BxDF::pdf(sd, woLocal, wiLocal, (int)sd.bsdfType);
				if (mGuiding.isEnableGuiding(w.depth))
					scatterPdf *= bsdfSamplingFraction;
			}
			if (mGuiding.isEnableGuiding(w.depth) && bsdfSamplingFraction < 1 &&
				(mGuiding.guideTransmissive || !(bsdfType & BSDF_TRANSMISSION))) {
				MixedSphericalGaussianDistribution<NUM_VMF_COMPONENTS> dist(dist_data);
				if (mGuiding.cosineAware && !(bsdfType & BSDF_TRANSMISSION))
					dist.applyCosineLobe(FaceForward(sd.frame.N, sd.wo));
				scatterPdf += dist.pdf(wiWorld) * (1 - bsdfSamplingFraction);
			}
			Color bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType);
			float misWeight = evalMIS(lightPdf, scatterPdf);
			
			if (misWeight > 0 && !isnan(misWeight) && !isinf(misWeight) && bsdfVal.any()) {
				ShadowRayWorkItem sw = {};
				sw.ray = shadowRay;
				sw.Li = ls.L;
				sw.a = w.thp * misWeight * bsdfVal * fabs(wiLocal[2]) / lightPdf;
				sw.pixelId = w.pixelId;
				sw.tMax = 1;
				if (sw.a.any()) shadowRayQueue->push(sw);
			}
		}
		
	
		if (mGuiding.isEnableGuiding(w.depth) && !(bsdfType & BSDF_SPECULAR) && 
			(mGuiding.guideTransmissive || !(bsdfType & BSDF_TRANSMISSION)) &&
			(bsdfSamplingFraction == 0 || sampler.get1D() >= bsdfSamplingFraction))
			guidedRayQueue->push(tid);
		else bsdfEvalQueue->push(tid);
	});
}

void GuidedPathTracer::handleBsdfSampling(){
	PROFILE("BSDF sampling");
	float* guidingOutput = inferenceOutputBuffer.data();
	ForAllQueued(bsdfEvalQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const BsdfEvalWorkItem & id) {
		const ScatterRayWorkItem w = scatterRayQueue->operator[](id.itemId);
		Sampler sampler = &pixelState->sampler[w.pixelId];
		const ShadingData& sd = w.sd;
		const BSDFType bsdfType = sd.getBsdfType();
		Vector3f woLocal = sd.frame.toLocal(sd.wo);

	
		BSDFSample sample = BxDF::sample(sd, woLocal, sampler, (int)sd.bsdfType);
		if (sample.pdf && any(sample.f)) {
			Vector3f wiWorld = sd.frame.toWorld(sample.wi);
			RayWorkItem r = {};
			Vector3f p = offsetRayOrigin(sd.pos, sd.frame.N, wiWorld);
			float bsdfPdf	 = sample.pdf, guidedPdf{};
			float scatterPdf = bsdfPdf;
			
			float *dist_data		   = guidingOutput + id.itemId * N_DIM_OUTPUT;
			float bsdfSamplingFraction = mGuiding.bsdfSamplingFraction;
#if GUIDED_LEARN_SELECTION
			bsdfSamplingFraction = 1 - 
				MixedSGWithSelection<NUM_VMF_COMPONENTS>::extractSelectionProbability(dist_data);
#endif
			
			
			if (mGuiding.isEnableGuiding(w.depth) && bsdfSamplingFraction < 1 
				&& !(bsdfType & BSDF_SPECULAR)
				&& (mGuiding.guideTransmissive || !(bsdfType & BSDF_TRANSMISSION))) {
				MixedSphericalGaussianDistribution<NUM_VMF_COMPONENTS> dist(dist_data);
				if (mGuiding.cosineAware && !(bsdfType & BSDF_TRANSMISSION)) 
					dist.applyCosineLobe(FaceForward(sd.frame.N, sd.wo));
				guidedPdf  = dist.pdf(wiWorld);
				scatterPdf = bsdfPdf * bsdfSamplingFraction + (1 - bsdfSamplingFraction) * guidedPdf;
			}
			r.pdf = scatterPdf;
			r.ray = { p, wiWorld };
			r.ctx = { sd.pos, sd.frame.N };
			r.pixelId = w.pixelId;
			r.depth = w.depth + 1;
			r.thp = w.thp * sample.f * fabs(sample.wi[2]) / r.pdf;
			r.bsdfType = sample.flags;
			nextRayQueue(w.depth)->push(r);
		
			if (mGuiding.isEnableTraining(w.depth) && mGuiding.isTrainingPixel(w.pixelId))
				guidedState->incrementDepth(w.pixelId, 
					Ray{sd.pos, wiWorld}, r.thp, scatterPdf, bsdfPdf,
					mGuiding.misAware ? (1 - bsdfSamplingFraction) * guidedPdf / scatterPdf : 1, 
					0, sample.isSpecular(),
					mGuiding.productWithCosine ?sample.f * fabs(sample.wi[2]) : sample.f, 
					sd);
		}
	});
}

void GuidedPathTracer::hendleGuidedSampling() {
	PROFILE("Guided sampling");
	float* output = inferenceOutputBuffer.data();

	ForAllQueued(guidedRayQueue, MAX_INFERENCE_NUM,
		KRR_DEVICE_LAMBDA(const GuidedInferenceWorkItem & w){
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		uint wid = w.itemId;
		const ScatterRayWorkItem& sw = scatterRayQueue->operator[](wid);
		const ShadingData& sd = sw.sd;
		const BSDFType bsdfType = sd.getBsdfType();

		Sampler sampler = &pixelState->sampler[sw.pixelId];
		float* dist_data = output + wid * N_DIM_OUTPUT;
		MixedSphericalGaussianDistribution<NUM_VMF_COMPONENTS> dist(dist_data);
		if (mGuiding.cosineAware && !(bsdfType & BSDF_TRANSMISSION))
			dist.applyCosineLobe(FaceForward(sd.frame.N, sd.wo));
		Vector3f wi = dist.sample(sampler);
		float bsdfSamplingFraction = mGuiding.bsdfSamplingFraction;
#if GUIDED_LEARN_SELECTION
		bsdfSamplingFraction = 1 - 
			MixedSGWithSelection<NUM_VMF_COMPONENTS>::extractSelectionProbability(dist_data);
#endif
		if (wi.hasNaN()) {
			printf("Found NaN in sampled guiding distribution.\n");
			return;
		} 
		Vector3f wiLocal = sd.frame.toLocal(wi);
		Vector3f woLocal = sd.frame.toLocal(sd.wo);

		Vector3f p = offsetRayOrigin(sd.pos, sd.frame.N, wi);
		Color bsdfVal = BxDF::f(sd, woLocal, wiLocal, (int)sd.bsdfType);
		float guidedPdf	 = dist.pdf(wi);
		float bsdfPdf = BxDF::pdf(sd, woLocal, wiLocal, (int) sd.bsdfType);
		float scatterPdf = (1 - bsdfSamplingFraction) * guidedPdf + bsdfSamplingFraction * bsdfPdf;

		
		RayWorkItem rw = {};
		rw.pdf = scatterPdf;
		rw.ray = { p, wi };
		rw.ctx = { sd.pos, sd.frame.N };
		rw.thp = sw.thp * bsdfVal * fabs(wiLocal[2]) / rw.pdf;
		rw.pixelId = sw.pixelId;
		rw.depth = sw.depth + 1;
		rw.bsdfType = BSDF_GLOSSY | (SameHemisphere(wiLocal, woLocal) ? BSDF_REFLECTION : BSDF_TRANSMISSION );
		if (rw.thp.any()) {	
			nextRayQueue(sw.depth)->push(rw);
		
			if (mGuiding.isEnableTraining(sw.depth) && mGuiding.isTrainingPixel(rw.pixelId))
				guidedState->incrementDepth(rw.pixelId, 
					Ray{ sd.pos, wi }, rw.thp, scatterPdf, bsdfPdf,
					mGuiding.misAware ? (1 - bsdfSamplingFraction) * guidedPdf / scatterPdf : 1, 
					0, false, 
					mGuiding.productWithCosine ? bsdfVal * fabs(wiLocal[2]) : bsdfVal,
					sd);
		}
	});
}

void GuidedPathTracer::generateCameraRays(int sampleId){
	PROFILE("Generate camera rays");
	RayQueue* cameraRayQueue = currentRayQueue(0);
	auto frameSize			 = getFrameSize();
	GPUParallelFor(
		maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId) {
			Sampler sampler		= &pixelState->sampler[pixelId];
			Vector2i pixelCoord = { pixelId % frameSize[0], pixelId / frameSize[0] };
			Ray cameraRay		= camera->getRay(pixelCoord, frameSize, sampler);
			cameraRayQueue->pushCameraRay(cameraRay, pixelId);
		});
}

void GuidedPathTracer::resize(const Vector2i& size){
	if (size[0] * size[1] > MAX_RESOLUTION) 
		Log(Fatal, "Currently maximum number of pixels is limited to %d", MAX_RESOLUTION);
	RenderPass::resize(size);
	initialize();		
}

void GuidedPathTracer::setScene(Scene::SharedPtr scene){
	scene->initializeSceneRT();
	mScene = scene;
	lightSampler = scene->mSceneRT->getSceneData().lightSampler;
	initialize();
	if (!backend) {
		backend = new OptixBackend();
		auto params = OptixInitializeParameters()
						  .setPTX(GUIDED_PTX)
						  .addRaygenEntry("Closest")
						  .addRaygenEntry("Shadow")
						  .addRayType("Closest", true, true, false)
						  .addRayType("Shadow", false, true, false);
		backend->initialize(params);
	}
	backend->setScene(scene);
}

void GuidedPathTracer::beginFrame(RenderContext *context) {
	if (!mScene || !maxQueueSize) return;
	PROFILE("Begin frame");
	auto frameSize = getFrameSize();
	cudaMemcpy(camera, &mScene->getCamera()->getCameraData(), sizeof(Camera::CameraData),
			   cudaMemcpyHostToDevice);
	GPUParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){	
		
			Vector2i pixelCoord	   = { pixelId % frameSize[0], pixelId / frameSize[0] };
		pixelState->L[pixelId] = 0;
		pixelState->sampler[pixelId].setPixelSample(pixelCoord, frameId * samplesPerPixel);
		pixelState->sampler[pixelId].advance(256 * pixelId);
		guidedState->reset(pixelId);
	});
	GPUCall(KRR_DEVICE_LAMBDA() { trainBuffer->clear(); });
	mGuiding.beginFrame();
	
	mGuiding.trainState.enableTraining =
		(mGuiding.isEnableTraining() || autoTrain) && !isTrainingFinished;
	mGuiding.trainState.enableGuiding = mGuiding.isEnableTraining() || autoTrain;
}

void GuidedPathTracer::render(RenderContext *context) {
	if (!mScene || !maxQueueSize) return;
	PROFILE("Guided Path Tracer");
	for (int sampleId = 0; sampleId < samplesPerPixel; sampleId++) {
		
		GPUCall(KRR_DEVICE_LAMBDA() { currentRayQueue(0)->reset(); });
		generateCameraRays(sampleId);
		
		for (int depth = 0; true; depth++) {
			GPUCall(KRR_DEVICE_LAMBDA() {
				nextRayQueue(depth)->reset();
				hitLightRayQueue->reset();
				missRayQueue->reset();
				shadowRayQueue->reset();
				scatterRayQueue->reset();
				bsdfEvalQueue->reset();
				guidedRayQueue->reset();
			});
			
			traceClosest(depth);
			
			handleEmissiveHit();
			handleMiss();
			
			if (depth == maxDepth) break;
			
			
			if (mGuiding.isEnableGuiding(depth)) 
				inferenceStep();
			
			handleIntersections();		
			if (enableNEE) traceShadow();
			
			if (mGuiding.isEnableGuiding(depth) && mGuiding.bsdfSamplingFraction < 1)
				hendleGuidedSampling();
			handleBsdfSampling();	
		}
	}
	task.tickFrame();
	float weight = 1.f;
	if (mGuiding.sampleWeighting) {
		weight = M_EPSILON + task.getProgress() / (task.getBudget().type == BudgetType::Spp
															 ? trainingBudgetSpp
															 : trainingBudgetTime);
		weight		 = min(pow(weight, 1.f), 1.f);
	}
	// Log(Info, "Weight is %f", weight);
	CudaRenderTarget cudaFrame = context->getColorTexture()->getCudaRenderTarget();
	GPUParallelFor(maxQueueSize, KRR_DEVICE_LAMBDA(int pixelId){
		Color L = pixelState->L[pixelId] / float(samplesPerPixel);
		if (enableClamp) L = clamp(L, 0.f, clampMax);
		renderedImage->put(Color4f(L, 1), weight, pixelId);
		if (renderMode == RenderMode::Interactive)
			cudaFrame.write(Color4f(L, 1), pixelId);
		else if (renderMode == RenderMode::Offline)
			cudaFrame.write(renderedImage->getPixel(pixelId), pixelId);
	});
}

void GuidedPathTracer::endFrame(RenderContext *context) {
	if (mGuiding.isEnableTraining() && !isTrainingFinished && !trainDebug) trainStep();
	if (trainDebug && oneStep) { trainStep(); oneStep = 0; }
	frameId++;
	
	Budget budget = task.getBudget();
	if ( autoTrain && !isTrainingFinished &&
		((budget.type == BudgetType::Spp && task.getProgress() >= trainingBudgetSpp) ||
		(budget.type == BudgetType::Time && task.getProgress() >= trainingBudgetTime))) {
		isTrainingFinished = true;
		cudaDeviceSynchronize();
		if (discardTraining)
			renderedImage->reset();
		CUDA_SYNC_CHECK();
	}
	if (task.isFinished())
		gpContext->requestExit();
}

void GuidedPathTracer::finalize() {
	
	cudaDeviceSynchronize();
	string output_name = gpContext->getGlobalConfig().contains("name") ? 
		gpContext->getGlobalConfig()["name"] : "result";
	fs::path save_path = File::outputDir() / (output_name + ".exr");
	renderedImage->save(save_path);
	Log(Info, "Total SPP: %zd, elapsed time: %.1f", task.getCurrentSpp(), task.getElapsedTime());
	Log(Success, "Task finished, saving results to %s", save_path.string().c_str());
	CUDA_SYNC_CHECK();
}

void GuidedPathTracer::Guidance::renderUI() {
	trainState.renderUI();
	ui::DragFloat("Bsdf sampling fraction", &bsdfSamplingFraction, 0.001f, 0, 1, "%.3f");
	ui::Checkbox("Cosine aware sampling", &cosineAware);
	static const char* loss_types[] = { "L1", "L2", "LogL1", "RelL2" };
	static const char* divergence_types[] = { "K-L", "Chi-Square" };
	static float lr = optimizer->learning_rate();
	if (ui::DragFloat("Learning rate", &lr, 1e-6f, 0, 1e-1, "%.6f")) {
		optimizer->set_learning_rate(lr);
	}
	if (ui::CollapsingHeader("Advanced training options")) {
		
		ui::Combo("Divergence type", (int*)&divergence_type, divergence_types, (int)EDivergence::NumTypes);
		if (ui::InputInt("Max guided depth", (int*)&maxGuidedDepth, 1))
			maxGuidedDepth = max(0U, (uint)maxGuidedDepth);
		if (ui::InputInt("Max train depth", (int*)&maxTrainDepth, 1))
			maxTrainDepth = max(0U, min(maxTrainDepth, (uint) MAX_TRAIN_DEPTH));
		if (ui::InputInt("Train pixel stride", (int*)&trainState.trainPixelStride, 1))
			trainState.trainPixelStride = max(1U, trainState.trainPixelStride);
		if (ui::InputInt("Train batch size", (int*)&batchSize, 1))
			batchSize = max(1U, min(batchSize, (uint)TRAIN_BATCH_SIZE));
		if (ui::InputInt("Batch per frame", (int*)&batchPerFrame, 1, 1))
			batchPerFrame = max(0U, batchPerFrame);
	}
	ui::Text("Current step: %d; %d samples; loss: %f", numLossSamples, numTrainingSamples, curLossScalar.emaVal());
	ui::PlotLines("Loss graph", lossGraph.data(), min(numLossSamples, lossGraph.size()),
		numLossSamples < LOSS_GRAPH_SIZE ? 0 : numLossSamples % LOSS_GRAPH_SIZE, 0, FLT_MAX, FLT_MAX, ImVec2(0, 50));
}

void GuidedPathTracer::renderUI(){
	ui::Text("Render parameters");
	ui::InputInt("Samples per pixel", &samplesPerPixel);
	ui::InputInt("Max bounces", &maxDepth, 1);
	ui::SliderFloat("Russian roulette", &probRR, 0, 1);
	ui::Checkbox("Enable NEE", &enableNEE);
	if (mGuiding.network) {	
		ui::Text("Guidance");
		mGuiding.renderUI();
	}
	ui::Text("Debugging");
	ui::Checkbox("Debug output", &debugOutput);
	if (debugOutput) {
		ui::SameLine();
		ui::InputInt("Debug pixel:", (int*) &debugPixel);
	}
	if (ui::CollapsingHeader("Task progress")) {
		task.renderUI();
	}
	ui::Checkbox("Train debug", &trainDebug); ui::SameLine();
	if (ui::Button("Train one step")) oneStep = 1;
	if (ui::Button("Reset parameters")) {
		resetTraining();
	}
	ui::Checkbox("Clamping pixel value", &enableClamp);
	if (enableClamp) {
		ui::SameLine();
		ui::DragFloat("Max:", &clampMax, 1, 1, 1e5f, "%.1f");
	}
}

void GuidedPathTracer::resetNetwork(json config){
	using namespace tcnn;
	mGuiding.config = config;
	json& encoding_config = config["encoding"];
	json& optimizer_config = config["optimizer"];
	json& network_config = config["network"];
	json& loss_config = config["loss"];			
	if (!mGuiding.stream) cudaStreamCreate(&mGuiding.stream);
	
	mGuiding.optimizer.reset(tcnn::create_optimizer<precision_t>(optimizer_config));
	mGuiding.encoding.reset(tcnn::create_encoding<precision_t>(N_DIM_INPUT, encoding_config));
	mGuiding.loss.reset(tcnn::create_loss<precision_t>(loss_config));
 	mGuiding.network = std::make_shared<GuidingNetwork<precision_t>>(
		mGuiding.encoding, N_DIM_OUTPUT, network_config);

	mGuiding.trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(
		mGuiding.network, mGuiding.optimizer, mGuiding.loss, KRR_DEFAULT_RND_SEED);

	Log(Info, "Network has a padded output width of %d", mGuiding.network->padded_output_width());
	CHECK_LOG(next_multiple(N_DIM_OUTPUT, 16u) == N_DIM_PADDED_OUTPUT, 
		"Padded network output width seems wrong!");
	CHECK_LOG(mGuiding.network->padded_output_width() == N_DIM_PADDED_OUTPUT,
		 "Padded network output width seems wrong!");
	trainOutputBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT * TRAIN_BATCH_SIZE);
	gradientBuffer = GPUMemory<precision_t>(N_DIM_PADDED_OUTPUT * TRAIN_BATCH_SIZE);
	lossBuffer = GPUMemory<float>(TRAIN_BATCH_SIZE);
	inferenceInputBuffer = GPUMemory<float>(N_DIM_INPUT * MAX_INFERENCE_NUM);
	inferenceOutputBuffer = GPUMemory<float>(N_DIM_OUTPUT * MAX_INFERENCE_NUM);
	
	mGuiding.trainer->initialize_params();
	mGuiding.sampler.setSeed(KRR_DEFAULT_RND_SEED);
	CUDA_SYNC_CHECK();
}

void GuidedPathTracer::resetTraining() {
	mGuiding.trainer->initialize_params();
	task.reset();
	numLossSamples = 0;
}


void GuidedPathTracer::inferenceStep(){
	PROFILE("Inference");
	cudaDeviceSynchronize();
	const cudaStream_t& stream = mGuiding.stream;
	std::shared_ptr<Network<float, precision_t>> network = mGuiding.network;
	if (!network) logFatal("Network not initialized!");
	int numInferenceSamples = scatterRayQueue->size();
	if (numInferenceSamples == 0) return;
	{
		PROFILE("Data preparation");
		LinearKernel(generate_inference_data, stream, MAX_INFERENCE_NUM,
			scatterRayQueue, inferenceInputBuffer.data(), mScene->getBoundingBox());
	}
	int paddedBatchSize = next_multiple(numInferenceSamples, 128);
	GPUMatrix<float> networkInputs(inferenceInputBuffer.data(), N_DIM_INPUT, paddedBatchSize);
	GPUMatrix<float> networkOutputs(inferenceOutputBuffer.data(), N_DIM_OUTPUT, paddedBatchSize);
	{
		PROFILE("Network inference");
		network->inference(stream, networkInputs, networkOutputs);
		CUDA_SYNC_CHECK();
	}
}

void GuidedPathTracer::trainStep(){
	PROFILE("Training");
	const cudaStream_t& stream = mGuiding.stream;
	std::shared_ptr<Network<float, precision_t>> network = mGuiding.network;
	if (!network) logFatal("Network not initialized!");
	uint numTrainPixels = maxQueueSize / mGuiding.trainState.trainPixelStride;
	LinearKernel(generate_training_data, stream, numTrainPixels,
		mGuiding.trainState.trainPixelOffset, mGuiding.trainState.trainPixelStride,
				 trainBuffer, guidedState, mScene->getBoundingBox());
	cudaDeviceSynchronize();
	numTrainingSamples = trainBuffer->size();
	
	uint numTrainBatches = min((uint)numTrainingSamples / mGuiding.batchSize + 1, mGuiding.batchPerFrame);
	for (int iter = 0; iter < numTrainBatches; iter++) {
		size_t localBatchSize = min(numTrainingSamples - iter * mGuiding.batchSize, (size_t)mGuiding.batchSize);
		localBatchSize -= localBatchSize % 128;
		if (localBatchSize < MIN_TRAIN_BATCH_SIZE) break;
		float* inputData = (float*)(trainBuffer->inputs() + iter * mGuiding.batchSize);
		GuidedOutput* outputData = trainBuffer->outputs() + iter * mGuiding.batchSize;

		GPUMatrix<float> networkInputs(inputData, N_DIM_INPUT, localBatchSize);
		GPUMatrix<precision_t> networkOutputs(trainOutputBuffer.data(), N_DIM_PADDED_OUTPUT, localBatchSize);
		GPUMatrix<precision_t> dL_doutput(gradientBuffer.data(), N_DIM_PADDED_OUTPUT, localBatchSize);
		
		{
			PROFILE("Train step");
			
			std::unique_ptr<tcnn::Context> ctx = network->forward(stream, networkInputs, &networkOutputs, false, false);
			
			CUDA_SYNC_CHECK();	
#if GUIDED_LEARN_SELECTION
			LinearKernel(compute_dL_doutput_with_selection, stream, localBatchSize,
						 networkOutputs.data(), outputData, dL_doutput.data(), lossBuffer.data(),
						 TRAIN_LOSS_SCALE, mGuiding.divergence_type);
#else
			LinearKernel(compute_dL_doutput_divergence, stream, localBatchSize,
				networkOutputs.data(), outputData, dL_doutput.data(), lossBuffer.data(), TRAIN_LOSS_SCALE, mGuiding.divergence_type);
#endif
			
			network->backward(stream, *ctx, networkInputs, networkOutputs, dL_doutput, nullptr, false, EGradientMode::Overwrite);
			mGuiding.trainer->optimizer_step(stream, TRAIN_LOSS_SCALE);
			
			float loss = thrust::reduce(thrust::device, lossBuffer.data(), lossBuffer.data() + localBatchSize, 0.f, thrust::plus<float>());
			curLossScalar.update(loss / localBatchSize);
			lossGraph[numLossSamples++ % LOSS_GRAPH_SIZE] = curLossScalar.emaVal();
		}
	}
}

KRR_REGISTER_PASS_DEF(GuidedPathTracer);
KRR_NAMESPACE_END