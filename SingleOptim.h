#pragma once

// Compute per-splat Lambertian albedo from SH samples.
// We solve for 'a' in the Lambertian equation: L_o = (a / PI) * E
// where E = L_i * cosTheta.
// To find the best 'a' across multiple samples (view directions), we use 
// the Least Squares estimator: a = PI * [ Sum(L_o * E) / Sum(E * E) ]

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "Math.h"
#include "BRDFSample.h"
#include <glm/glm.hpp>

inline std::vector<glm::vec3> optimizeDiffuseFromSamples2(const std::vector<BRDFSample>& samples, std::vector<Gaussian>& all,
	int Iterations = 0, // Set to 0 as Least-Squares provides the optimal solution directly
	float learningRate = 0.01f)
{
	// Group samples by splat index
	std::unordered_map<int, std::vector<const BRDFSample*>> groups;
	for (const auto& s : samples) {
		groups[s.splatIndex].push_back(&s);
	}

	size_t maxIndex = 0;
	for (const auto& kv : groups) {
		if ((size_t)kv.first > maxIndex) maxIndex = (size_t)kv.first;
	}

	// Default to a neutral gray
	std::vector<glm::vec3> albedos(maxIndex + 1, glm::vec3(0.5f));

	for (const auto& kv : groups) {
		int idx = kv.first;
		const auto& brdfSampleList = kv.second;

		// Numerator: Sum(L_obs * E)
		// Denominator: Sum(E * E)
		// Where E = L_i * cosTheta
		glm::vec3 numerator(0.0f);
		glm::vec3 denominator(0.0f);

		for (const BRDFSample* ps : brdfSampleList) {
			float cosTheta = std::max(0.0f, ps->cosTheta);

			// Incident Irradiance sample
			glm::vec3 E = ps->L_i * cosTheta;

			// Accumulate for per-channel Least Squares
			numerator.r += ps->L_o.r * E.r;
			numerator.g += ps->L_o.g * E.g;
			numerator.b += ps->L_o.b * E.b;

			denominator.r += E.r * E.r;
			denominator.g += E.g * E.g;
			denominator.b += E.b * E.b;
		}

		glm::vec3 finalAlbedo;
		const float epsilon = 1e-8f;

		// Solve: a = PI * (Numerator / Denominator)
		finalAlbedo.r = (denominator.r > epsilon) ? (numerator.r / denominator.r) * (float)M_PI : 0.5f;
		finalAlbedo.g = (denominator.g > epsilon) ? (numerator.g / denominator.g) * (float)M_PI : 0.5f;
		finalAlbedo.b = (denominator.b > epsilon) ? (numerator.b / denominator.b) * (float)M_PI : 0.5f;

		// Clamp to physically plausible range [0, 1]
		// Note: If albedo > 1.0, your SH is likely "hotter" (more specular) than Lambertian can represent
		finalAlbedo = glm::clamp(finalAlbedo, 0.0f, 1.0f);

		albedos[idx] = finalAlbedo;
	}

	// Write results to CSV
	std::ofstream out("albedos2.csv");
	out << "splatIndex,albedo.r,albedo.g,albedo.b,sampleCount\n";
	for (size_t i = 0; i < albedos.size(); ++i) {
		auto it = groups.find((int)i);
		if (it == groups.end()) continue;
		const glm::vec3& a = albedos[i];
		out << i << "," << a.x << "," << a.y << "," << a.z << "," << it->second.size() << "\n";
	}
	out.close();

	// Update the gaussians for rendering
	for (size_t i = 0; i < albedos.size(); i++) {
		if (i < all.size()) {
			all[i].testAlbedo = albedos[i];
		}
	}

	std::cout << "BRDF optimizer: Extracted optimal albedos for " << groups.size() << " splats.\n";
	return albedos;
}