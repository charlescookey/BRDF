#pragma once

// Simplified Disney BRDF Optimizer - Optimizes only the 4 core parameters
// This version is faster and easier to debug than the full 11-parameter version
//
// Parameters optimized:
// - baseColor (vec3): Surface color
// - metallic (float): 0 = dielectric, 1 = metallic
// - roughness (float): Surface roughness
// - specular (float): Specular strength (usually keep at 0.5)
//
// Changes from original:
// - Adam optimizer replaces vanilla SGD (fixes poor baseColor convergence)
// - Initialization from SH color as prior (warm start, not cold start from 0.5)
// - Per-parameter learning rates (baseColor needs higher LR than scalar params)
// - Gradient clipping to prevent exploding updates
// - Stagnation threshold loosened to allow slower but real progress

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


// -------------------------------------------------------------------------
// Adam optimizer state — one instance per splat
// -------------------------------------------------------------------------
struct AdamStateSimple {
    // First moment (mean)
    glm::vec3 m_bc = glm::vec3(0.0f);
    float     m_met = 0.0f;
    float     m_rough = 0.0f;
    float     m_spec = 0.0f;

    // Second moment (uncentered variance)
    glm::vec3 v_bc = glm::vec3(0.0f);
    float     v_met = 0.0f;
    float     v_rough = 0.0f;
    float     v_spec = 0.0f;

    int t = 0; // Step counter (bias correction)

    static constexpr float beta1 = 0.9f;
    static constexpr float beta2 = 0.999f;
    static constexpr float epsilon = 1e-8f;

    // Apply one Adam step and return the bias-corrected update for a vec3 param.
    glm::vec3 stepVec3(const glm::vec3& grad, float lr) {
        m_bc = beta1 * m_bc + (1.0f - beta1) * grad;
        v_bc = beta2 * v_bc + (1.0f - beta2) * (grad * grad);
        glm::vec3 m_hat = m_bc / (1.0f - std::pow(beta1, (float)t));
        glm::vec3 v_hat = v_bc / (1.0f - std::pow(beta2, (float)t));
        return lr * m_hat / (glm::sqrt(v_hat) + epsilon);
    }

    // Apply one Adam step and return the bias-corrected update for a scalar param.
    float stepScalar(float grad, float& m, float& v, float lr) {
        m = beta1 * m + (1.0f - beta1) * grad;
        v = beta2 * v + (1.0f - beta2) * (grad * grad);
        float m_hat = m / (1.0f - std::pow(beta1, (float)t));
        float v_hat = v / (1.0f - std::pow(beta2, (float)t));
        return lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
};

namespace DisneySimple {

    inline float schlickFresnel(float u) {
        float m = glm::clamp(1.0f - u, 0.0f, 1.0f);
        return m * m * m * m * m; // m^5
    }

    inline float GTR2(float NdotH, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
        return a2 / (float(M_PI) * t * t);
    }

    inline float smithG_GGX(float NdotV, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float b = NdotV * NdotV;
        return 1.0f / (NdotV + std::sqrt(a2 + b - a2 * b));
    }

    // Simplified Disney diffuse (Burley)
    inline float disneyDiffuse(float NdotL, float NdotV, float LdotH, float roughness) {
        float fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
        float FL = schlickFresnel(NdotL);
        float FV = schlickFresnel(NdotV);
        return (1.0f + (fd90 - 1.0f) * FL) * (1.0f + (fd90 - 1.0f) * FV);
    }

    // Core Disney BRDF evaluation (diffuse + specular only)
    inline glm::vec3 evaluate(
        const DisneyBRDFParamsSimple& p,
        const glm::vec3& V,   // View direction  (unit)
        const glm::vec3& L,   // Light direction (unit)
        const glm::vec3& N    // Surface normal  (unit)
    ) {
        float NdotL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
        float NdotV = glm::clamp(glm::dot(N, V), 0.0f, 1.0f);

        if (NdotL <= 0.0f || NdotV <= 0.0f) return glm::vec3(0.0f);

        glm::vec3 H = glm::normalize(L + V);
        float     NdotH = glm::clamp(glm::dot(N, H), 0.0f, 1.0f);
        float     LdotH = glm::clamp(glm::dot(L, H), 0.0f, 1.0f);

        // Specular base color: dielectric uses white tinted by specular scalar,
        // metallic uses baseColor directly.
        glm::vec3 Cspec0 = glm::mix(
            p.specular * 0.08f * glm::vec3(1.0f), // Dielectric F0
            p.baseColor,                           // Metallic F0
            p.metallic
        );

        // Diffuse
        float     Fd = disneyDiffuse(NdotL, NdotV, LdotH, p.roughness);
        float     invPI = 1.0f / float(M_PI);
        glm::vec3 diffuse = p.baseColor * invPI * Fd;

        // Specular (GGX)
        float     D = GTR2(NdotH, p.roughness);
        float     FH = schlickFresnel(LdotH);
        glm::vec3 F = glm::mix(Cspec0, glm::vec3(1.0f), FH);
        float     G = smithG_GGX(NdotL, p.roughness) * smithG_GGX(NdotV, p.roughness);
        glm::vec3 spec = G * F * D;

        // Combine: attenuate diffuse for metals
        return (1.0f - p.metallic) * diffuse + spec;
    }

} // namespace DisneySimple

// -------------------------------------------------------------------------
// Finite-difference gradient computation
// -------------------------------------------------------------------------
void computeGradient(
    const DisneyBRDFParamsSimple& p,
    const BRDFSample& sample,
    glm::vec3& grad_baseColor,
    float& grad_metallic,
    float& grad_roughness,
    float& grad_specular,
    float epsilon = 1e-4f
) {
    // Baseline loss
    glm::vec3 f0 = DisneySimple::evaluate(p, sample.omega_o, sample.omega_i, sample.normal);
    glm::vec3 pred0 = f0 * sample.L_i * sample.cosTheta;
    glm::vec3 residual = pred0 - sample.L_o;
    float     loss0 = glm::dot(residual, residual);

    DisneyBRDFParamsSimple temp = p;

    // --- baseColor (per channel) ---
    grad_baseColor = glm::vec3(0.0f);
    for (int i = 0; i < 3; ++i) {
        float orig = temp.baseColor[i];
        temp.baseColor[i] = orig + epsilon;

        glm::vec3 f1 = DisneySimple::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
        glm::vec3 res = f1 * sample.L_i * sample.cosTheta - sample.L_o;
        float loss1 = glm::dot(res, res);

        grad_baseColor[i] = (loss1 - loss0) / epsilon;
        temp.baseColor[i] = orig;
    }

    // --- metallic ---
    temp.metallic = p.metallic + epsilon;
    {
        glm::vec3 f1 = DisneySimple::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
        glm::vec3 res = f1 * sample.L_i * sample.cosTheta - sample.L_o;
        grad_metallic = (glm::dot(res, res) - loss0) / epsilon;
    }
    temp.metallic = p.metallic;

    // --- roughness ---
    temp.roughness = p.roughness + epsilon;
    {
        glm::vec3 f1 = DisneySimple::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
        glm::vec3 res = f1 * sample.L_i * sample.cosTheta - sample.L_o;
        grad_roughness = (glm::dot(res, res) - loss0) / epsilon;
    }
    temp.roughness = p.roughness;

    // --- specular ---
    temp.specular = p.specular + epsilon;
    {
        glm::vec3 f1 = DisneySimple::evaluate(temp, sample.omega_o, sample.omega_i, sample.normal);
        glm::vec3 res = f1 * sample.L_i * sample.cosTheta - sample.L_o;
        grad_specular = (glm::dot(res, res) - loss0) / epsilon;
    }
}

// -------------------------------------------------------------------------
// Main simplified Disney BRDF optimizer
//
// Key changes vs original:
//   - Adam replaces SGD  ?  handles different gradient scales per param
//   - SH colour used as warm-start for baseColor  ?  much closer initial value
//   - Per-parameter LR multipliers  ?  baseColor needs 10x more than scalars
//   - Gradient clipping  ?  prevents blow-up on high-intensity samples
//   - Stagnation window widened to 30  ?  Adam needs longer to settle
// -------------------------------------------------------------------------
void optimizeDisneyBRDFSimple(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>& gaussians,
    int   maxIterations = 500,
    float learningRate = 0.01f,   // Base LR — higher than SGD default is fine for Adam
    bool  verbose = true
) {
    // Per-parameter LR multipliers (tuned empirically)
    constexpr float LR_MULT_BASECOLOR = 1.0f;   // Full LR for colour
    constexpr float LR_MULT_METALLIC = 0.1f;   // Scalar params need less
    constexpr float LR_MULT_ROUGHNESS = 0.1f;
    constexpr float LR_MULT_SPECULAR = 0.1f;

    // Gradient clipping threshold (per-parameter, before Adam scaling)
    constexpr float GRAD_CLIP = 10.0f;

    if (verbose) {
        std::cout << "=== Simplified Disney BRDF Optimizer (Adam) ===\n";
        std::cout << "Max iterations : " << maxIterations << "\n";
        std::cout << "Base LR        : " << learningRate << "\n";
        std::cout << "LR baseColor   : " << learningRate * LR_MULT_BASECOLOR << "\n";
        std::cout << "LR scalars     : " << learningRate * LR_MULT_METALLIC << "\n";
    }

    // Group samples by splat index
    std::unordered_map<int, std::vector<const BRDFSample*>> groups;
    for (const auto& s : samples)
        groups[s.splatIndex].push_back(&s);

    size_t maxIndex = 0;
    for (const auto& kv : groups)
        maxIndex = std::max<size_t>(maxIndex, (size_t)kv.first);

    std::vector<DisneyBRDFParamsSimple> params(maxIndex + 1);

    // -----------------------------------------------------------------------
    // Warm-start baseColor from SH (DC radiance), clamp to valid albedo range.
    // SH colour encodes outgoing radiance so it's a biased but close proxy for
    // albedo — far better than always starting at grey (0.5, 0.5, 0.5).
    // -----------------------------------------------------------------------
    for (size_t i = 0; i < params.size() && i < gaussians.size(); ++i) {
        params[i].baseColor = glm::clamp(gaussians[i].testColor, 0.02f, 0.98f);
        params[i].metallic = 0.0f;
        params[i].roughness = 0.5f;
        params[i].specular = 0.5f;
    }
    // Any splat beyond gaussians.size() falls back to neutral grey
    for (size_t i = gaussians.size(); i < params.size(); ++i) {
        params[i].baseColor = glm::vec3(0.5f);
    }

    if (verbose) {
        std::cout << "Optimizing " << groups.size() << " splats with "
            << samples.size() << " total samples...\n";
    }

    std::vector<AdamStateSimple> adamStates(maxIndex + 1);

    int progressInterval = std::max(1, (int)groups.size() / 10);
    int processed = 0;

    // -----------------------------------------------------------------------
    // Per-splat optimization loop
    // -----------------------------------------------------------------------
    for (const auto& [splatIdx, sampleList] : groups) {
        if (sampleList.empty()) continue;

        DisneyBRDFParamsSimple& p = params[splatIdx];
        AdamStateSimple& adam = adamStates[splatIdx];

        float prevLoss = 1e10f;
        int   stagnantCount = 0;

        for (int iter = 0; iter < maxIterations; ++iter) {

            glm::vec3 totalGrad_bc = glm::vec3(0.0f);
            float     totalGrad_met = 0.0f;
            float     totalGrad_rough = 0.0f;
            float     totalGrad_spec = 0.0f;
            float     totalLoss = 0.0f;

            // Accumulate gradients over all samples for this splat
            for (const BRDFSample* s : sampleList) {
                glm::vec3 g_bc;
                float     g_m, g_r, g_s;
                computeGradient(p, *s, g_bc, g_m, g_r, g_s);

                totalGrad_bc += g_bc;
                totalGrad_met += g_m;
                totalGrad_rough += g_r;
                totalGrad_spec += g_s;

                glm::vec3 f = DisneySimple::evaluate(p, s->omega_o, s->omega_i, s->normal);
                glm::vec3 pred = f * s->L_i * s->cosTheta;
                glm::vec3 res = pred - s->L_o;
                totalLoss += glm::dot(res, res);
            }

            // Average over samples
            float n = static_cast<float>(sampleList.size());
            totalGrad_bc /= n;
            totalGrad_met /= n;
            totalGrad_rough /= n;
            totalGrad_spec /= n;
            totalLoss /= n;

            // Gradient clipping (element-wise for vec3, magnitude for scalars)
            totalGrad_bc = glm::clamp(totalGrad_bc, glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
            totalGrad_met = glm::clamp(totalGrad_met, -GRAD_CLIP, GRAD_CLIP);
            totalGrad_rough = glm::clamp(totalGrad_rough, -GRAD_CLIP, GRAD_CLIP);
            totalGrad_spec = glm::clamp(totalGrad_spec, -GRAD_CLIP, GRAD_CLIP);

            // Adam update — increment step counter once per iteration
            adam.t++;

            p.baseColor -= adam.stepVec3(totalGrad_bc, learningRate * LR_MULT_BASECOLOR);
            p.metallic -= adam.stepScalar(totalGrad_met, adam.m_met, adam.v_met, learningRate * LR_MULT_METALLIC);
            p.roughness -= adam.stepScalar(totalGrad_rough, adam.m_rough, adam.v_rough, learningRate * LR_MULT_ROUGHNESS);
            p.specular -= adam.stepScalar(totalGrad_spec, adam.m_spec, adam.v_spec, learningRate * LR_MULT_SPECULAR);

            p.clamp();

            // Early stopping — converged
            if (totalLoss < 1e-6f) break;

            // Early stopping — stagnation (widened window for Adam)
            if (std::abs(prevLoss - totalLoss) < 1e-8f) {
                if (++stagnantCount > 30) break;
            }
            else {
                stagnantCount = 0;
            }
            prevLoss = totalLoss;
        }

        ++processed;
        if (verbose && processed % progressInterval == 0) {
            std::cout << "Progress: " << (processed * 100 / (int)groups.size()) << "%\r" << std::flush;
        }
    }

    if (verbose) std::cout << "\nOptimization complete!\n";

    // -----------------------------------------------------------------------
    // Write CSV results
    // -----------------------------------------------------------------------
    std::ofstream out("disney_brdf_simple.csv");
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

    // Update Gaussian albedo
    //for (size_t i = 0; i < params.size() && i < gaussians.size(); ++i)
      //  gaussians[i].testAlbedo = params[i].baseColor;

    if (verbose) std::cout << "Results saved to disney_brdf_simple.csv\n";
}