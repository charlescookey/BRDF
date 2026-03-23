#pragma once

// Disney BRDF Optimizer autodiff edition
//
// Uses the autodiff library (https://autodiff.github.io) in REVERSE MODE
// (autodiff::var) to compute analytic gradients of the MSE loss w.r.t.
// the four core Disney BRDF parameters:
//
//   baseColor  (vec3)    surface albedo
//   metallic   (float)   0 = dielectric, 1 = metallic
//   roughness  (float)   surface roughness
//   specular   (float)   specular weight (dielectric F0 scale)
//
// Why autodiff instead of finite differences?
//
//   Finite-difference (old):   
//   Autodiff reverse-mode:     1 forward pass + 1 backward pass per sample
//                              exact gradients, no E tuning, ~3-6× faster
// 
//
// Dependencies:
//   autodiff   header-only, install via vcpkg: `vcpkg install autodiff`
//               or CMake FetchContent from https://github.com/autodiff/autodiff
//   glm        already used by original code
//   C++17 or later
//
// Adam optimizer and all hyper-parameters are unchanged from the original.

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "Math.h"
#include "BRDFSample.h"

#include <glm/glm.hpp>

// autodiff reverse-mode header  (var.hpp only no Eigen integration needed)
#include <autodiff/reverse/var.hpp>

using namespace autodiff;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


//
// Adam optimizer state  (identical to the original)
// 
struct AdamStateAD {
    // First moments
    glm::vec3 m_bc = glm::vec3(0.f);
    float     m_met = 0.f, m_rough = 0.f, m_spec = 0.f;

    // Second moments
    glm::vec3 v_bc = glm::vec3(0.f);
    float     v_met = 0.f, v_rough = 0.f, v_spec = 0.f;

    int t = 0;

    static constexpr float beta1 = 0.9f;
    static constexpr float beta2 = 0.999f;
    static constexpr float epsilon = 1e-8f;

    glm::vec3 stepVec3(const glm::vec3& grad, float lr) {
        m_bc = beta1 * m_bc + (1.f - beta1) * grad;
        v_bc = beta2 * v_bc + (1.f - beta2) * (grad * grad);
        glm::vec3 mh = m_bc / (1.f - std::pow(beta1, (float)t));
        glm::vec3 vh = v_bc / (1.f - std::pow(beta2, (float)t));
        return lr * mh / (glm::sqrt(vh) + epsilon);
    }

    float stepScalar(float grad, float& m, float& v, float lr) {
        m = beta1 * m + (1.f - beta1) * grad;
        v = beta2 * v + (1.f - beta2) * (grad * grad);
        float mh = m / (1.f - std::pow(beta1, (float)t));
        float vh = v / (1.f - std::pow(beta2, (float)t));
        return lr * mh / (std::sqrt(vh) + epsilon);
    }
};


//
// Templated Disney BRDF  (works with T = float  OR  T = autodiff::var)
// 
namespace DisneyAD {

    // Helper: Schlick Fresnel  (works for any scalar type)
    template<typename T>
    inline T schlickFresnel(T u) {
        T m = T(1.0) - u;
        if (m < T(0.0)) m = T(0.0);
        if (m > T(1.0)) m = T(1.0);
        return m * m * m * m * m;
    }

    // GGX NDF
    template<typename T>
    inline T GTR2(T NdotH, T roughness) {
        T a = roughness * roughness;
        T a2 = a * a;
        T t = T(1.0) + (a2 - T(1.0)) * NdotH * NdotH;
        return a2 / (T(M_PI) * t * t);
    }

    // Smith GGX masking term
    template<typename T>
    inline T smithG_GGX(T NdotV, T roughness) {
        T a = roughness * roughness;
        T a2 = a * a;
        T b = NdotV * NdotV;
        return T(1.0) / (NdotV + sqrt(a2 + b - a2 * b));
    }

    // Burley diffuse
    template<typename T>
    inline T disneyDiffuse(T NdotL, T NdotV, T LdotH, T roughness) {
        T fd90 = T(0.5) + T(2.0) * LdotH * LdotH * roughness;
        T FL = schlickFresnel(NdotL);
        T FV = schlickFresnel(NdotV);
        return (T(1.0) + (fd90 - T(1.0)) * FL) * (T(1.0) + (fd90 - T(1.0)) * FV);
    }

    // -------------------------------------------------------------------------
    // Core BRDF evaluation
    //
    // Params are passed as individual autodiff::var scalars so the tape can
    // track them.  Directions and normal remain plain glm::vec3 (no gradient
    // needed through geometry).
    // -------------------------------------------------------------------------
    template<typename T>
    inline void evaluate(
        // BRDF parameters (differentiable when T = autodiff::var)
        T bc_r, T bc_g, T bc_b,
        T metallic,
        T roughness,
        T specular,
        // Geometry (plain floats no differentiation through these)
        const glm::vec3& V,
        const glm::vec3& L,
        const glm::vec3& N,
        // Outputs: three colour channels
        T& out_r, T& out_g, T& out_b
    ) {
        // Dot products with clamping
        float ndl_f = glm::clamp(glm::dot(N, L), 0.f, 1.f);
        float ndv_f = glm::clamp(glm::dot(N, V), 0.f, 1.f);

        if (ndl_f <= 0.f || ndv_f <= 0.f) {
            out_r = out_g = out_b = T(0.0);
            return;
        }

        glm::vec3 H = glm::normalize(L + V);
        float ndh_f = glm::clamp(glm::dot(N, H), 0.f, 1.f);
        float ldh_f = glm::clamp(glm::dot(L, H), 0.f, 1.f);

        T NdotL = T(ndl_f);
        T NdotV = T(ndv_f);
        T NdotH = T(ndh_f);
        T LdotH = T(ldh_f);

        // Specular base colour:  lerp(specular*0.08, baseColor, metallic)
        T Cspec_r = (T(1.0) - metallic) * specular * T(0.08) + metallic * bc_r;
        T Cspec_g = (T(1.0) - metallic) * specular * T(0.08) + metallic * bc_g;
        T Cspec_b = (T(1.0) - metallic) * specular * T(0.08) + metallic * bc_b;

        // Diffuse
        T Fd = disneyDiffuse(NdotL, NdotV, LdotH, roughness);
        T invPI = T(1.0 / M_PI);
        T diff_r = bc_r * invPI * Fd;
        T diff_g = bc_g * invPI * Fd;
        T diff_b = bc_b * invPI * Fd;

        // Specular
        T D = GTR2(NdotH, roughness);
        T FH = schlickFresnel(LdotH);
        T F_r = Cspec_r + (T(1.0) - Cspec_r) * FH;
        T F_g = Cspec_g + (T(1.0) - Cspec_g) * FH;
        T F_b = Cspec_b + (T(1.0) - Cspec_b) * FH;
        T G = smithG_GGX(NdotL, roughness) * smithG_GGX(NdotV, roughness);

        T spec_r = G * F_r * D;
        T spec_g = G * F_g * D;
        T spec_b = G * F_b * D;

        // Combine
        out_r = (T(1.0) - metallic) * diff_r + spec_r;
        out_g = (T(1.0) - metallic) * diff_g + spec_g;
        out_b = (T(1.0) - metallic) * diff_b + spec_b;
    }

} // namespace DisneyAD


//
// Gradient computation via autodiff reverse mode
//
// Builds the MSE loss for one BRDFSample over autodiff::var params,
// then calls autodiff::gradient() to get dloss/dparams in one backward pass.
//
// Returns the scalar loss value (as float) for convergence tracking.
//
struct BRDFGradients {
    glm::vec3 bc;       // dL/dbaseColor  (per channel)
    float     metallic;
    float     roughness;
    float     specular;
    float     loss;     // MSE value (not differentiated)
};

BRDFGradients computeGradientAD(
    const DisneyBRDFParamsSimple& p,
    const BRDFSample& s
) {
    // Declare differentiable parameters on the tape
    var bc_r(p.baseColor.r), bc_g(p.baseColor.g), bc_b(p.baseColor.b);
    var met(p.metallic), rough(p.roughness), spec(p.specular);

    // Forward pass: BRDF output channels
    var fr, fg, fb;
    DisneyAD::evaluate(bc_r, bc_g, bc_b, met, rough, spec,
        s.omega_o, s.omega_i, s.normal,
        fr, fg, fb);

    // Predicted outgoing radiance  L_pred = f * L_i * cosTheta
    /*var pred_r = fr * val(s.L_i.r) * val(s.cosTheta);
    var pred_g = fg * val(s.L_i.g) * val(s.cosTheta);
    var pred_b = fb * val(s.L_i.b) * val(s.cosTheta);*/
    
    var pred_r = fr * val(s.L_i.r) ;
    var pred_g = fg * val(s.L_i.g) ;
    var pred_b = fb * val(s.L_i.b) ;

    // Residuals
    var res_r = pred_r - val(s.L_o.r);
    var res_g = pred_g - val(s.L_o.g);
    var res_b = pred_b - val(s.L_o.b);

    // Scalar MSE loss
    var loss = res_r * res_r + res_g * res_g + res_b * res_b;

    // Backward pass
    // derivatives(loss, wrt(...)) does one reverse sweep and returns
    // an array of doubles  one per variable listed in wrt().
    // No Eigen, no seed/propagate/grad needed.
    auto [d_bc_r, d_bc_g, d_bc_b, d_met, d_rough, d_spec] =
        derivatives(loss, wrt(bc_r, bc_g, bc_b, met, rough, spec));

    BRDFGradients out;
    out.bc.r = static_cast<float>(d_bc_r);
    out.bc.g = static_cast<float>(d_bc_g);
    out.bc.b = static_cast<float>(d_bc_b);
    out.metallic = static_cast<float>(d_met);
    out.roughness = static_cast<float>(d_rough);
    out.specular = static_cast<float>(d_spec);
    out.loss = static_cast<float>(val(loss));
    return out;
}


//
// Main optimizer  (drop-in replacement for optimizeDisneyBRDFSimple)
// 
void optimizeDisneyBRDFAutodiff(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>& gaussians,
    int   maxIterations = 500,
    float learningRate = 0.01f,
    bool  verbose = true
) {
    // Per-parameter LR multipliers (same rationale as original)
    constexpr float LR_BC = 1.0f;
    constexpr float LR_MET = 0.1f;
    constexpr float LR_ROUGH = 0.1f;
    constexpr float LR_SPEC = 0.1f;

    // Gradient clipping threshold
    constexpr float GRAD_CLIP = 10.0f;

    if (verbose) {
        std::cout << "=== Disney BRDF Optimizer (autodiff reverse-mode) ===\n";
        std::cout << "Max iterations : " << maxIterations << "\n";
        std::cout << "Base LR        : " << learningRate << "\n";
        std::cout << "Gradient source: analytic (reverse-mode AD)\n";
    }

    //  Group samples by splat
    std::unordered_map<int, std::vector<const BRDFSample*>> groups;
    for (const auto& s : samples)
        groups[s.splatIndex].push_back(&s);

    size_t maxIndex = 0;
    for (const auto& [idx, _] : groups)
        maxIndex = std::max<size_t>(maxIndex, (size_t)idx);

    std::vector<DisneyBRDFParamsSimple> params(maxIndex + 1);

    // Warm-start from SH colour
    for (size_t i = 0; i < params.size() && i < gaussians.size(); ++i) {
        params[i].baseColor = glm::clamp(gaussians[i].testColor, 0.02f, 0.98f);
        params[i].metallic = 0.f;
        params[i].roughness = 0.5f;
        params[i].specular = 0.5f;
    }
    for (size_t i = gaussians.size(); i < params.size(); ++i)
        params[i].baseColor = glm::vec3(0.5f);

    if (verbose) {
        std::cout << "Optimizing " << groups.size() << " splats, "
            << samples.size() << " total samples\n";
    }

    std::vector<AdamStateAD> adam(maxIndex + 1);

    int progressInterval = std::max(1, (int)groups.size() / 10);
    int processed = 0;

    // Per-splat optimization loop
    for (const auto& [splatIdx, sampleList] : groups) {
        if (sampleList.empty()) continue;

        DisneyBRDFParamsSimple& p = params[splatIdx];
        AdamStateAD& ad = adam[splatIdx];

        float prevLoss = 1e10f;
        int   stagnantCount = 0;

        for (int iter = 0; iter < maxIterations; ++iter) {

            glm::vec3 totalGrad_bc = glm::vec3(0.f);
            float     totalGrad_met = 0.f;
            float     totalGrad_rough = 0.f;
            float     totalGrad_spec = 0.f;
            float     totalLoss = 0.f;

            // Accumulate gradients (analytic, one reverse pass per sample)
            for (const BRDFSample* s : sampleList) {
                BRDFGradients g = computeGradientAD(p, *s);
                totalGrad_bc += g.bc;
                totalGrad_met += g.metallic;
                totalGrad_rough += g.roughness;
                totalGrad_spec += g.specular;
                totalLoss += g.loss;
            }

            // Average
            float n = static_cast<float>(sampleList.size());
            totalGrad_bc /= n;
            totalGrad_met /= n;
            totalGrad_rough /= n;
            totalGrad_spec /= n;
            totalLoss /= n;

            // Gradient clipping
            totalGrad_bc = glm::clamp(totalGrad_bc,
                glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
            totalGrad_met = glm::clamp(totalGrad_met, -GRAD_CLIP, GRAD_CLIP);
            totalGrad_rough = glm::clamp(totalGrad_rough, -GRAD_CLIP, GRAD_CLIP);
            totalGrad_spec = glm::clamp(totalGrad_spec, -GRAD_CLIP, GRAD_CLIP);

            // Adam update
            ad.t++;
            p.baseColor -= ad.stepVec3(totalGrad_bc, learningRate * LR_BC);
            p.metallic -= ad.stepScalar(totalGrad_met, ad.m_met, ad.v_met, learningRate * LR_MET);
            p.roughness -= ad.stepScalar(totalGrad_rough, ad.m_rough, ad.v_rough, learningRate * LR_ROUGH);
            p.specular -= ad.stepScalar(totalGrad_spec, ad.m_spec, ad.v_spec, learningRate * LR_SPEC);

            p.clamp();

            // Early stop converged
            if (totalLoss < 1e-6f) break;

            // Early stop – stagnation
            if (std::abs(prevLoss - totalLoss) < 1e-8f) {
                if (++stagnantCount > 30) break;
            }
            else {
                stagnantCount = 0;
            }
            prevLoss = totalLoss;
        }

        ++processed;
        if (verbose && processed % progressInterval == 0)
            std::cout << "Progress: " << (processed * 100 / (int)groups.size())
            << "%\r" << std::flush;
    }

    if (verbose) std::cout << "\nOptimization complete!\n";

    //Write CSV
    std::ofstream out("disney_brdf_autodiff.csv");
    out << "splatIndex,baseColor.r,baseColor.g,baseColor.b,"
        "metallic,roughness,specular,sampleCount,SH.r,SH.g,SH.b\n";

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
            << gaussians[i].color.r << ","
            << gaussians[i].color.g << ","
            << gaussians[i].color.b << "\n";
    }
    out.close();

    if (verbose) std::cout << "Results saved to disney_brdf_autodiff.csv\n";
}