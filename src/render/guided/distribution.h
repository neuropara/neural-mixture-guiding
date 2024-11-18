#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "train.h"
#include "util/math_utils.h"
#include "util/vmf.h"
#include "common.h"

KRR_NAMESPACE_BEGIN

class VMFKernel {
public:

	VMFKernel() = default;
	
	KRR_CALLABLE VMFKernel(float lambda, float kappa, float theta, float phi) :
		lambda(lambda), kappa(kappa), theta(theta), phi(phi) {}

	KRR_CALLABLE Vector2f getSphericalDir() const { return { theta, phi }; }
	
	KRR_CALLABLE Vector3f getCartesianDir() const { 
		return utils::sphericalToCartesian(theta, phi);
	}
	
	KRR_CALLABLE Vector3f sample(Vector2f u) const {
		return VMFDistribution(kappa).sample(u, theta, phi);
	}

	KRR_CALLABLE float eval(const Vector3f& wi) const {
		return lambda * pdf(wi);
	}
	
	KRR_CALLABLE float pdf(const Vector3f& wi) const {
		return VMFDistribution(kappa).eval(wi, theta, phi);
	}

	KRR_CALLABLE void product(const VMFKernel& other) {
		auto norm = [](const float kappa) {
			return kappa < minValidKappa ? M_INV_4PI : kappa * M_INV_2PI / (1 - expf(-2 * kappa));
		};
		
		Vector3f mu_a = getCartesianDir(), mu_b = other.getCartesianDir();
		Vector3f new_mu = mu_a * kappa + mu_b * other.kappa;
		float new_kappa = length(new_mu);

		if (new_kappa >= 1e-3)	
			new_mu /= new_kappa;
		else {			
			new_kappa = 0;
			new_mu = mu_a;
		}
		Vector2f new_mu_sph = utils::cartesianToSpherical(new_mu);
		
		
		float e = expf(kappa * (new_mu.dot(mu_a) - 1) + other.kappa * (new_mu.dot(mu_b) - 1));
		float s = norm(kappa) * norm(other.kappa) / norm(new_kappa) * e;
		
		float new_lambda = s * lambda * other.lambda;
		lambda = new_lambda;
		kappa = new_kappa;
		theta = new_mu_sph[0], phi = new_mu_sph[1];
	}
	
	
	static constexpr float minValidKappa = 1e-3f;
	
	float lambda;		
	float kappa;		
	float theta;		
	float phi;			
};

template <int N>
class MixedSphericalGaussianDistribution {
public:

	enum {N_COMP = N};

	MixedSphericalGaussianDistribution() = default;
	
	template <typename T>
	
	KRR_CALLABLE MixedSphericalGaussianDistribution (T* data) {
		totalWeight = 0;
		for(int i = 0; i < N; i++) {
			int idx = i * N_DIM_VMF;
			mSG[i].lambda = network_to_params((float)data[idx], ACTIVATION_LAMBDA);
			mSG[i].kappa = network_to_params((float)data[idx + 1], ACTIVATION_KAPPA);
			mSG[i].theta = (float)M_PI* network_to_params((float)data[idx + 2], ACTIVATION_THETA);
			mSG[i].phi = (float)M_2PI* network_to_params((float)data[idx + 3], ACTIVATION_PHI);
			totalWeight += mSG[i].lambda;
		}
		DCHECK_NE(totalWeight, 0);
		for(int i = 0; i < N; i++)
			weight[i] = mSG[i].lambda / totalWeight;
	}

	KRR_CALLABLE float eval(Vector3f wi) const {
		float val = 0.0f;
		for (int i = 0; i < N; i++)
			val += mSG[i].eval(wi);
		return val;
	}

	
	KRR_CALLABLE float eval(uint i, Vector3f wi) const {
		DCHECK_LT(i, N);
		return mSG[i].eval(wi);
	}
	
	KRR_CALLABLE float pdf(Vector3f wi) const {
		DCHECK(wi.any());
		float pdf = 0.0f;
		for (int i = 0; i < N; i++)
			pdf += weight[i] * mSG[i].pdf(wi);
		return pdf;
	}

	
	KRR_CALLABLE float pdf(uint i, Vector3f wi) const {
		DCHECK_LT(i, N);
		return mSG[i].pdf(wi);
	}

	KRR_CALLABLE const VMFKernel &get(uint i) {
		DCHECK_LT(i, N);
		return mSG[i];
	}

	KRR_CALLABLE Vector3f sample(Sampler& sampler) const {
		float u = sampler.get1D();
		for (int i = 0; i < N; i++) { 
			if (u < weight[i]) 
				return mSG[i].sample(sampler.get2D());
			u -= weight[i];
		}
		return mSG[0].sample(sampler.get2D());
	}


	template <typename T>
	KRR_CALLABLE float gradients_probability(const Vector3f wi, T* output) const {
		float x = wi[0], y = wi[1], z = wi[2];
		float probability = 0;		
		for (int sg = 0; sg < N; sg++) {
			precision_t* cur_gradient = output + sg * 4;

			float lambda = mSG[sg].lambda, kappa = mSG[sg].kappa,
				theta = mSG[sg].theta, phi = mSG[sg].phi;
			
			float vmf = pdf(sg, wi);
			probability += weight[sg] * vmf;
			
			float dF_dlambda = vmf * (totalWeight - lambda) / pow2(totalWeight);
			for (int k = 0; k < N; k++) {
				if (k != sg)
					dF_dlambda -= weight[k] / totalWeight * mSG[k].pdf(wi);
			}

			float inv_kappa_minus_inv_tanh_kappa;
			// Workaround for numerical instability issue when kappa is small (possibly leading to NaNs).
			// Thanks @tyanyuy3125 for this fix!
			if (kappa < 1.f)
				inv_kappa_minus_inv_tanh_kappa = 0.000926f + -0.344883f * kappa + 0.030147f * pow2(kappa);
			else 
				inv_kappa_minus_inv_tanh_kappa = 1 / kappa - (1 + expf(-2 * kappa)) / (1 - expf(-2 * kappa));
		
			float dF_dkappa = weight[sg] * vmf
				* (x * sin(theta) * cos(phi) + y * sin(theta) * sin(phi) + z * cos(theta) + inv_kappa_minus_inv_tanh_kappa);
		
			float dF_dtheta = weight[sg] * vmf * kappa * (cos(theta) * (x * cos(phi) + y * sin(phi)) - z * sin(theta));
		
			float dF_dphi = weight[sg] * vmf * kappa * (y * cos(phi) - x * sin(phi)) * sin(theta);

			cur_gradient[0] = dF_dlambda, cur_gradient[1] = dF_dkappa;
			cur_gradient[2] = dF_dtheta, cur_gradient[3] = dF_dphi;
		}
		return probability;
	}
	

	KRR_CALLABLE void applyCosineLobe(Vector3f normal) {
		Vector2f normal_sph = utils::cartesianToSpherical(normal);
		VMFKernel vmf_cosine = VMFKernel(1, VMF_DIFFUSE_LOBE, normal_sph[0], normal_sph[1]);
		totalWeight = 0;
		for (int i = 0; i < N; i++) {
			mSG[i].product(vmf_cosine);
			totalWeight += mSG[i].lambda;
		}
		totalWeight = max(totalWeight, M_EPSILON);
		for (int i = 0; i < N; i++)
			weight[i] = mSG[i].lambda / totalWeight;
	}

	static constexpr float VMF_DIFFUSE_LOBE = 2.18853f;

protected:
	VMFKernel mSG[N];
	float weight[N], totalWeight{};
};

template <int N> 
class MixedSGWithSelection : public MixedSphericalGaussianDistribution<N> {
public:
	
	MixedSGWithSelection() = default;
	
	template <typename T>
	
	KRR_CALLABLE MixedSGWithSelection(T* data)
		:MixedSphericalGaussianDistribution<N>(data) {
		selectionProb = extractSelectionProbability(data);
	}

	KRR_CALLABLE float pdf(const Vector3f &wi, float bsdfPdf) {
		return selectionProb * MixedSphericalGaussianDistribution<N>::pdf(wi) +
			   (1 - selectionProb) * bsdfPdf;
	}

	template <typename T>
	KRR_CALLABLE float gradients_probability(const Vector3f wi, const float bsdfPdf, T *output) const {
		float probability =
			MixedSphericalGaussianDistribution<N>::gradients_probability(wi, output);
		for (int i = 0; i < N * N_DIM_VMF; i++) 
			output[i] = (float) output[i] * selectionProb;
		output[N * N_DIM_VMF] = T(probability - bsdfPdf);
		return probability;
	}

	KRR_CALLABLE float getSelectionProbability() { return selectionProb; }

	template <typename T> 
	KRR_CALLABLE static float extractSelectionProbability(T* data) {
		return (float) network_to_params((float) data[N * N_DIM_VMF], ACTIVATION_PROB);
	}

protected:
	float selectionProb{};
};

KRR_NAMESPACE_END