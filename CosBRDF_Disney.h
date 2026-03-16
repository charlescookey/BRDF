#pragma once

// Disney BRDF Optimizer with COSINE SIMILARITY Loss
// 
// Instead of L2 loss: ||predicted - observed||²
// Uses cosine similarity: 1 - (predicted · observed) / (||predicted|| × ||observed||)
//
// Benefits:
// - Focuses on COLOR DIRECTION (hue/ratios) rather than brightness
// - More robust to lighting intensity variations
// - Better for matching relative color without worrying about absolute intensity
//
// Parameters optimized:
// - baseColor (vec3): Surface color
// - metallic (float): 0 = dielectric, 1 = metallic
// - roughness (float): Surface roughness
// - specular (float): Specular strength

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "Math.h"
#include "BRDFSample.h"
#include <glm/glm.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


namespace DisneyCosineSim {

    inline float schlickFresnel(float u) {
        float m = glm::clamp(1.0f - u, 0.0f, 1.0f);
        return m * m * m * m * m;
    }

    inline float GTR2(float NdotH, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
        return a2 / (M_PI * t * t);
    }

    inline float smithG_GGX(float NdotV, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float b = NdotV * NdotV;
        return 1.0f / (NdotV + std::sqrt(a2 + b - a2 * b));
    }

    inline float disneyDiffuse(float NdotL, float NdotV, float LdotH, float roughness) {
        float fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
        float FL = schlickFresnel(NdotL);
        float FV = schlickFresnel(NdotV);
        return (1.0f + (fd90 - 1.0f) * FL) * (1.0f + (fd90 - 1.0f) * FV);
    }

    glm::vec3 evaluate(
        const DisneyBRDFParamsSimple& p,
        const glm::vec3& V,
        const glm::vec3& L,
        const glm::vec3& N
    ) {
        float NdotL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
        float NdotV = glm::clamp(glm::dot(N, V), 0.0f, 1.0f);

        if (NdotL <= 0.0f || NdotV <= 0.0f) return glm::vec3(0.0f);

        glm::vec3 H = glm::normalize(L + V);
        float NdotH = glm::clamp(glm::dot(N, H), 0.0f, 1.0f);
        float LdotH = glm::clamp(glm::dot(L, H), 0.0f, 1.0f);

        float lum = 0.3f * p.baseColor.r + 0.6f * p.baseColor.g + 0.1f * p.baseColor.b;
        glm::vec3 Ctint = lum > 0.0f ? p.baseColor / lum : glm::vec3(1.0f);

        glm::vec3 Cspec0 = glm::mix(
            p.specular * 0.08f * glm::vec3(1.0f),
            p.baseColor,
            p.metallic
        );

        float Fd = disneyDiffuse(NdotL, NdotV, LdotH, p.roughness);
        float     invPI = 1.0f / float(M_PI);
        glm::vec3 diffuse = p.baseColor * invPI * Fd;

        float D = GTR2(NdotH, p.roughness);
        float FH = schlickFresnel(LdotH);
        glm::vec3 F = glm::mix(Cspec0, glm::vec3(1.0f), FH);
        float G = smithG_GGX(NdotL, p.roughness) * smithG_GGX(NdotV, p.roughness);

        glm::vec3 specular = G * F * D;

        return (1.0f - p.metallic) * diffuse + specular;
    }
}

// Cosine similarity: similarity = (a · b) / (||a|| × ||b||)
// Loss = 1 - similarity (want to minimize, so we want high similarity)
float cosineSimilarityLoss(const glm::vec3& a, const glm::vec3& b) {
    float dot_ab = glm::dot(a, b);
    float norm_a = glm::length(a);
    float norm_b = glm::length(b);

    const float eps = 1e-8f;
    if (norm_a < eps || norm_b < eps) {
        return 1.0f; // Maximum dissimilarity for zero vectors
    }

    float cosine_sim = dot_ab / (norm_a * norm_b);
    // Clamp to [-1, 1] for numerical stability
    cosine_sim = glm::clamp(cosine_sim, -1.0f, 1.0f);

    // Loss = 1 - similarity (range [0, 2], 0 = perfect match)
    return 1.0f - cosine_sim;
}

// Gradient of cosine similarity loss w.r.t. predicted vector
glm::vec3 cosineSimilarityGradient(const glm::vec3& pred, const glm::vec3& obs) {
    float norm_pred = glm::length(pred);
    float norm_obs = glm::length(obs);

    const float eps = 1e-8f;
    if (norm_pred < eps || norm_obs < eps) {
        return glm::vec3(0.0f);
    }

    float dot_product = glm::dot(pred, obs);

    // ?Loss/?pred = -?(cosine_sim)/?pred
    // where cosine_sim = (pred · obs) / (||pred|| × ||obs||)
    //
    // Using quotient rule:
    // ?(cosine_sim)/?pred = [obs × ||pred|| - pred × (pred·obs)/||pred||] / (||pred||² × ||obs||)
    //                      = [obs / (||pred|| × ||obs||)] - [pred × (pred·obs) / (||pred||³ × ||obs||)]
    //                      = obs / (||pred|| × ||obs||) - pred × cosine_sim / ||pred||²

    float inv_norm_pred = 1.0f / norm_pred;
    float inv_norm_obs = 1.0f / norm_obs;
    float cosine_sim = dot_product * inv_norm_pred * inv_norm_obs;

    glm::vec3 grad_cosine = (obs * inv_norm_pred * inv_norm_obs) -
        (pred * cosine_sim * inv_norm_pred * inv_norm_pred);

    // Gradient of loss = -gradient of similarity
    return -grad_cosine;
}

// Compute gradient using finite differences with cosine similarity loss
void computeGradientCosineSim(
    const DisneyBRDFParamsSimple& p,
    const BRDFSample& sample,
    glm::vec3& grad_baseColor,
    float& grad_metallic,
    float& grad_roughness,
    float& grad_specular,
    float epsilon = 1e-4f
) {
    // Current prediction
    glm::vec3 f0 = DisneyCosineSim::evaluate(p, sample.omega_o, sample.omega_i, sample.normal);
    glm::vec3 pred0 = f0 * sample.L_i * sample.cosTheta;
    float loss0 = cosineSimilarityLoss(pred0, sample.L_o);

    DisneyBRDFParamsSimple temp = p;

    // Gradient for baseColor (per channel)
    grad_baseColor = glm::vec3(0.0f);
    for (int i = 0; i < 3; ++i) {
        float orig = temp.baseColor[i];
        temp.baseColor[i] = orig + epsilon;

        glm::vec3 f1 = DisneyCosineSim::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
        glm::vec3 pred1 = f1 * sample.L_i * sample.cosTheta;
        float loss1 = cosineSimilarityLoss(pred1, sample.L_o);

        grad_baseColor[i] = (loss1 - loss0) / epsilon;
        temp.baseColor[i] = orig;
    }

    // Gradient for metallic
    temp.metallic = p.metallic + epsilon;
    glm::vec3 f1 = DisneyCosineSim::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
    glm::vec3 pred1 = f1 * sample.L_i * sample.cosTheta;
    float loss1 = cosineSimilarityLoss(pred1, sample.L_o);
    grad_metallic = (loss1 - loss0) / epsilon;
    temp.metallic = p.metallic;

    // Gradient for roughness
    temp.roughness = p.roughness + epsilon;
    f1 = DisneyCosineSim::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
    pred1 = f1 * sample.L_i * sample.cosTheta;
    loss1 = cosineSimilarityLoss(pred1, sample.L_o);
    grad_roughness = (loss1 - loss0) / epsilon;
    temp.roughness = p.roughness;

    // Gradient for specular
    temp.specular = p.specular + epsilon;
    f1 = DisneyCosineSim::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
    pred1 = f1 * sample.L_i * sample.cosTheta;
    loss1 = cosineSimilarityLoss(pred1, sample.L_o);
    grad_specular = (loss1 - loss0) / epsilon;
}

// Disney BRDF optimizer with COSINE SIMILARITY loss
void optimizeDisneyBRDFCosineSim(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>& gaussians,
    int maxIterations = 500,
    float learningRate = 0.001f,
    bool verbose = true
) {
    if (verbose) {
        std::cout << "=== Disney BRDF Optimizer (COSINE SIMILARITY Loss) ===\n";
        std::cout << "Loss function: 1 - cos(predicted, observed)\n";
        std::cout << "Benefit: Matches COLOR DIRECTION, not brightness\n";
        std::cout << "Max iterations: " << maxIterations << "\n";
        std::cout << "Learning rate: " << learningRate << "\n";
    }

    // Group samples
    std::unordered_map<int, std::vector<const BRDFSample*>> groups;
    for (const auto& s : samples) {
        groups[s.splatIndex].push_back(&s);
    }

    size_t maxIndex = 0;
    for (const auto& kv : groups) {
        maxIndex = std::max<size_t>(maxIndex, (size_t)kv.first);
    }

    std::vector<DisneyBRDFParamsSimple> params(maxIndex + 1);

    // Initialize
    for (size_t i = 0; i < params.size() && i < gaussians.size(); ++i) {
        params[i].baseColor = glm::clamp(gaussians[i].testColor, 0.02f, 0.98f);
        params[i].metallic = 0.0f;
        params[i].roughness = 0.5f;
        params[i].specular = 0.5f;
    }

    if (verbose) {
        std::cout << "Optimizing " << groups.size() << " splats...\n";
    }

    int progressInterval = std::max(1, (int)groups.size() / 10);
    int processed = 0;

    for (const auto& [splatIdx, sampleList] : groups) {
        if (sampleList.empty()) continue;

        DisneyBRDFParamsSimple& p = params[splatIdx];

        float prevLoss = 1e10f;
        int stagnantCount = 0;

        for (int iter = 0; iter < maxIterations; ++iter) {
            glm::vec3 totalGrad_baseColor(0.0f);
            float totalGrad_metallic = 0.0f;
            float totalGrad_roughness = 0.0f;
            float totalGrad_specular = 0.0f;
            float totalLoss = 0.0f;

            // Accumulate gradients
            for (const BRDFSample* sample : sampleList) {
                glm::vec3 grad_bc;
                float grad_m, grad_r, grad_s;

                computeGradientCosineSim(p, *sample, grad_bc, grad_m, grad_r, grad_s);

                totalGrad_baseColor += grad_bc;
                totalGrad_metallic += grad_m;
                totalGrad_roughness += grad_r;
                totalGrad_specular += grad_s;

                // Compute loss for monitoring
                glm::vec3 f = DisneyCosineSim::evaluate(p, sample->omega_o, sample->omega_i, sample->normal);
                glm::vec3 pred = f * sample->L_i * sample->cosTheta;
                totalLoss += cosineSimilarityLoss(pred, sample->L_o);
            }

            // Average
            float n = static_cast<float>(sampleList.size());
            totalGrad_baseColor /= n;
            totalGrad_metallic /= n;
            totalGrad_roughness /= n;
            totalGrad_specular /= n;
            totalLoss /= n;

            // Update
            p.baseColor -= learningRate * totalGrad_baseColor;
            p.metallic -= learningRate * totalGrad_metallic;
            p.roughness -= learningRate * totalGrad_roughness;
            p.specular -= learningRate * totalGrad_specular;

            p.clamp();

            // Early stopping (cosine similarity loss is in [0, 2])
            if (totalLoss < 0.001f) break; // Very high similarity

            if (std::abs(prevLoss - totalLoss) < 1e-8f) {
                stagnantCount++;
                if (stagnantCount > 10) break;
            }
            else {
                stagnantCount = 0;
            }
            prevLoss = totalLoss;
        }

        processed++;
        if (verbose && processed % progressInterval == 0) {
            std::cout << "Progress: " << (processed * 100 / groups.size()) << "%\r" << std::flush;
        }
    }

    if (verbose) std::cout << "\nOptimization complete!\n";

    // Write results
    std::ofstream out("disney_brdf_simple_cos_2.csv");
    out << "splatIndex,baseColor.r,baseColor.g,baseColor.b,metallic,roughness,specular,sampleCount,SH.r,SH.g,SH.b\n";

    for (size_t i = 0; i < params.size(); ++i) {
        auto it = groups.find((int)i);
        if (it == groups.end()) continue;

        const DisneyBRDFParamsSimple& p = params[i];
        out << i << ","
            << p.baseColor.r << "," << p.baseColor.g << "," << p.baseColor.b << ","
            << p.metallic << ","
            << p.roughness << ","
            << p.specular << ","
            << it->second.size() << ","
            << gaussians[i].color.r << "," << gaussians[i].color.g << "," << gaussians[i].color.b << "\n";
    }
    out.close();

    // Update gaussians
    for (size_t i = 0; i < params.size() && i < gaussians.size(); ++i) {
        gaussians[i].testAlbedo = params[i].baseColor;
    }

    if (verbose) {
        std::cout << "Results saved to disney_brdf_cosine.csv\n";
        std::cout << "Note: Cosine similarity focuses on color DIRECTION/HUE\n";
        std::cout << "      not absolute brightness. Great for matching tints!\n";
    }
}
