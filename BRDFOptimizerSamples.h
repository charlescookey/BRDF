#pragma once

// Compute per-splat Lambertian albedo from an in-memory BRDF sample list.
// Uses E_sample = L_i * cosTheta and gradient derived:
//   L_o_pred = (a / PI) * sum_s E_sample_s
//   dL_o_pred/da = (1 / PI) * sum_s E_sample_s
//   dLoss/da = 2 * (L_o_pred - L_o_obs) * (1 / PI) * sum_s E_sample_s

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "Math.h"            // Vec3, Colour, Gaussian
#include "BRDFSample.h"
#include <glm/glm.hpp>

void optimizeDiffuseFromSamples(const std::vector<BRDFSample>& samples, std::vector<Gaussian>& all,
	int Iterations = 0,
	float learningRate = 0.01f)
{
	// Group samples by splat index
	std::unordered_map<int, std::vector<const BRDFSample*>> groups;
	for (const auto& s : samples) {
		groups[s.splatIndex].push_back(&s);
	}

	size_t maxIndex = 0;
	for (const auto& kv : groups) maxIndex = std::max<size_t>(maxIndex, (size_t)kv.first);
	std::vector<glm::vec3> albedos(maxIndex + 1, glm::vec3(0.5f));

	struct SplatStats { glm::vec3 S; glm::vec3 meanLo; size_t count; };
	std::unordered_map<int, SplatStats> stats;
	const double EPS = 1e-12;

	// First pass: compute closed-form albedo and store stats (S = sum L_i * cosTheta, meanLo)
	for (const auto& kv : groups) {
		int idx = kv.first;
		const auto& brdfSampleList = kv.second;

		glm::vec3 SumLCosTheta(0.0f);
		glm::vec3 sumLo(0.0f);
		for (const BRDFSample* ps : brdfSampleList) {
			glm::vec3 E = ps->L_i * ps->cosTheta; // use cosTheta weighting as in derivation
			SumLCosTheta += E;
			sumLo += ps->L_o;
		}
		glm::vec3 meanLo = sumLo / static_cast<float>(brdfSampleList.size());

		// cache stats for reuse
		stats[idx] = { SumLCosTheta, meanLo, brdfSampleList.size() };

		// Closed-form per-channel a = PI * (S · meanLo) / (S · S)
		double ss_r = static_cast<double>(SumLCosTheta.x) * static_cast<double>(SumLCosTheta.x);
		double ss_g = static_cast<double>(SumLCosTheta.y) * static_cast<double>(SumLCosTheta.y);
		double ss_b = static_cast<double>(SumLCosTheta.z) * static_cast<double>(SumLCosTheta.z);

		glm::vec3 a;
		a.x = (float)(M_PI * (static_cast<double>(SumLCosTheta.x) * static_cast<double>(meanLo.x)) / (ss_r + EPS));
		a.y = (float)(M_PI * (static_cast<double>(SumLCosTheta.y) * static_cast<double>(meanLo.y)) / (ss_g + EPS));
		a.z = (float)(M_PI * (static_cast<double>(SumLCosTheta.z) * static_cast<double>(meanLo.z)) / (ss_b + EPS));

		// fallback if SumLCosTheta is nearly zero
		if (ss_r < 1e-12) a.x = 0.5f;
		if (ss_g < 1e-12) a.y = 0.5f;
		if (ss_b < 1e-12) a.z = 0.5f;

		// clamp to [0,1]
		a.x = std::max(0.0f, std::min(1.0f, a.x));
		a.y = std::max(0.0f, std::min(1.0f, a.y));
		a.z = std::max(0.0f, std::min(1.0f, a.z));

		albedos[idx] = a;
	}


	if (Iterations > 0) {
		for (const auto& kv : groups) {
			int idx = kv.first;
			glm::vec3 a = albedos[idx];

			// fetch precomputed stats
			const SplatStats& st = stats[idx];
			const glm::vec3& S = st.S;
			const glm::vec3& meanLo = st.meanLo;

			for (int it = 0; it < Iterations; ++it) {
				// pred = (a / PI) * S
				glm::vec3 pred = (a / (float)M_PI) * S;
				glm::vec3 residual = pred - meanLo;

				// gradient: dLoss/da = 2 * residual * (1 / PI) * SumLCosTheta
				glm::vec3 grad = 2.0f * (residual * (S / (float)M_PI));

				// gradient descent step
				a -= learningRate * grad;

				// clamp
				a.x = std::max(0.0f, std::min(1.0f, a.x));
				a.y = std::max(0.0f, std::min(1.0f, a.y));
				a.z = std::max(0.0f, std::min(1.0f, a.z));
			}
			albedos[idx] = a;
		}
	}

	// Write results
	
	std::ofstream out("albedos.csv");
	out << "splatIndex,albedo.r,albedo.g,albedo.b,sampleCount,SH.r,SH.g,SH.b\n";
	for (size_t i = 0; i < albedos.size(); ++i) {
		auto it = groups.find((int)i);
		if (it == groups.end()) continue;
		const glm::vec3& a = albedos[i];
		out << i << "," << a.x << "," << a.y << "," << a.z << "," << it->second.size() << ","
		<< all[i].color.r << "," << all[i].color.g << "," << all[i].color.b << "\n";
	}
	out.close();
	


	//update the gaussians
	for (size_t i = 0; i < albedos.size(); i++) {
		all[i].testAlbedo = albedos[i];
	}

	//std::cout << "BRDF optimizer: wrote albedos.csv (" << albedos.size() << " entries)\n";
}