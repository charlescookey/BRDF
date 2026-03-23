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
// Loss formulation (fixed):
//
//   Instead of treating each hemisphere sample as an independent (Li -> Lo)
//   pair, we compute the full Monte Carlo estimate of the rendering equation
//   per splat and compare it against the SH colour (the true Lo):
//
//     L_o_pred = (pi / N) * sum_i[ BRDF(wi, wo, N) * Li * NdotL_i ]
//     loss     = || L_o_pred - L_o_sh ||^2
//
//   This is the correct formulation because L_o_sh IS the integral of the
//   rendering equation -- the SH encodes the view-dependent appearance of
//   the splat, which is exactly what BRDF * Li integrated over the hemisphere
//   should equal.
//
// Dependencies:
//   autodiff   header-only, install via vcpkg: `vcpkg install autodiff`
//               or CMake FetchContent from https://github.com/autodiff/autodiff
//   glm        already used by original code
//   C++17 or later

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "Math.h"
#include "BRDFSample.h"

#include <glm/glm.hpp>

// autodiff reverse-mode header (var.hpp only, no Eigen integration needed)
#include <autodiff/reverse/var.hpp>

using namespace autodiff;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


//
// Adam optimizer state (identical to the original)
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
// Templated Disney BRDF (works with T = float OR T = autodiff::var)
//
namespace DisneyAD {

    // Schlick Fresnel
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
    // Directions and normal are plain glm::vec3 (no gradient through geometry).
    // omega_i MUST be normalized before calling -- caller's responsibility.
    // -------------------------------------------------------------------------
    template<typename T>
    inline void evaluate(
        T bc_r, T bc_g, T bc_b,
        T metallic,
        T roughness,
        T specular,
        const glm::vec3& V,   // view direction (omega_o), normalized
        const glm::vec3& L,   // light direction (omega_i), normalized
        const glm::vec3& N,   // surface normal, normalized
        T& out_r, T& out_g, T& out_b
    ) {
        float ndl_f = glm::clamp(glm::dot(N, L), 0.f, 1.f);
        float ndv_f = glm::clamp(glm::dot(N, V), 0.f, 1.f);

        if (ndl_f <= 0.f || ndv_f <= 0.f) {
            out_r = out_g = out_b = T(0.0);
            return;
        }

        glm::vec3 H = glm::normalize(L + V);
        float     ndh_f = glm::clamp(glm::dot(N, H), 0.f, 1.f);
        float     ldh_f = glm::clamp(glm::dot(L, H), 0.f, 1.f);

        T NdotL = T(ndl_f);
        T NdotV = T(ndv_f);
        T NdotH = T(ndh_f);
        T LdotH = T(ldh_f);

        // Specular base colour: lerp(specular*0.08, baseColor, metallic)
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

        // Combine diffuse + specular (metallic suppresses diffuse)
        out_r = (T(1.0) - metallic) * diff_r + spec_r;
        out_g = (T(1.0) - metallic) * diff_g + spec_g;
        out_b = (T(1.0) - metallic) * diff_b + spec_b;
    }

} // namespace DisneyAD


//
// Gradient struct (unchanged)
//
struct BRDFGradients {
    glm::vec3 bc;
    float     metallic;
    float     roughness;
    float     specular;
    float     loss;
};


// -------------------------------------------------------------------------
// computeGradientAD
//
// Takes ALL samples for one splat and computes a single gradient via the
// Monte Carlo rendering equation estimate.
//
// For cosine-weighted hemisphere sampling (PDF = NdotL/pi), the estimator is:
//
//   L_o_pred = (pi / N) * sum_i[ BRDF(wi, wo, N, params) * Li ]
//              (NdotL cancels with the PDF, so it does NOT appear in the sum)
//   loss     = || L_o_pred - L_o_sh ||^2
//
// L_o_sh is taken from samples[0]->L_o -- it is the SH colour of the splat
// evaluated in the camera direction, which is constant across all samples
// for the same splat.
//
// omega_i is normalized here before use (fixes the unnormalized direction bug).
// Samples where N dot omega_i <= 0 are culled (back-hemisphere).
// L_i components are clamped to [0, 2] (prevents negative SH radiance).
// -------------------------------------------------------------------------
BRDFGradients computeGradientAD(
    const DisneyBRDFParamsSimple& p,
    const std::vector<const BRDFSample*>& samples
) {
    var bc_r(p.baseColor.r), bc_g(p.baseColor.g), bc_b(p.baseColor.b);
    var met(p.metallic), rough(p.roughness), spec(p.specular);

    // MC integral for cosine-weighted hemisphere sampling.
    //
    // The rendering equation is:
    //   Lo = integral[ BRDF(wi,wo) * Li(wi) * NdotL ] dwi
    //
    // For cosine-weighted sampling the PDF is:  PDF(wi) = NdotL / pi
    //
    // The MC estimator is:
    //   Lo ~= (1/N) * sum[ BRDF * Li * NdotL / PDF ]
    //       = (1/N) * sum[ BRDF * Li * NdotL / (NdotL/pi) ]
    //       = (pi/N) * sum[ BRDF * Li ]       <-- NdotL cancels with PDF
    //
    // DO NOT multiply by NdotL inside the sum -- it is already accounted
    // for by the cosine-weighted sampling distribution.
    var sum_r(0.0), sum_g(0.0), sum_b(0.0);
    int validCount = 0;

    for (const BRDFSample* s : samples) {
        // Normalize omega_i
        float len = glm::length(s->omega_i);
        if (len < 1e-6f) continue;
        glm::vec3 wi = s->omega_i / len;

        // Cull back-hemisphere samples
        float ndl = glm::dot(s->normal, wi);
        if (ndl <= 0.f) continue;

        // Clamp Li to physically plausible range
        float li_r = glm::clamp(s->L_i.r, 0.f, 2.f);
        float li_g = glm::clamp(s->L_i.g, 0.f, 2.f);
        float li_b = glm::clamp(s->L_i.b, 0.f, 2.f);

        var fr, fg, fb;
        DisneyAD::evaluate(bc_r, bc_g, bc_b, met, rough, spec,
            s->omega_o, wi, s->normal,
            fr, fg, fb);

        // Accumulate BRDF * Li only -- NdotL cancels with cosine-sampling PDF
        sum_r += fr * val(li_r);
        sum_g += fg * val(li_g);
        sum_b += fb * val(li_b);

        ++validCount;
    }

    if (validCount == 0) {
        return BRDFGradients{ glm::vec3(0.f), 0.f, 0.f, 0.f, 0.f };
    }

    // (pi/N) * sum[ BRDF * Li ]
    double n = static_cast<double>(validCount);
    var pred_r = sum_r * (M_PI / n);
    var pred_g = sum_g * (M_PI / n);
    var pred_b = sum_b * (M_PI / n);

    // L_o target: SH colour of this splat from the camera direction.
    // This IS the correct target -- it encodes the view-dependent appearance
    // which equals the rendering equation integral by definition.
    // All samples for the same splat share the same L_o, so take from [0].
    var res_r = pred_r - val(samples[0]->L_o.r);
    var res_g = pred_g - val(samples[0]->L_o.g);
    var res_b = pred_b - val(samples[0]->L_o.b);

    var loss = res_r * res_r + res_g * res_g + res_b * res_b;

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
// Main optimizer (drop-in replacement for optimizeDisneyBRDFSimple)
//
void optimizeDisneyBRDFAutodiff(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>& gaussians,
    int   maxIterations = 500,
    float learningRate = 0.01f,
    bool  verbose = true
) {
    constexpr float LR_BC = 1.0f;
    constexpr float LR_MET = 0.1f;
    constexpr float LR_ROUGH = 0.1f;
    constexpr float LR_SPEC = 0.1f;
    constexpr float GRAD_CLIP = 10.0f;

    if (verbose) {
        std::cout << "=== Disney BRDF Optimizer (autodiff reverse-mode) ===\n";
        std::cout << "Max iterations : " << maxIterations << "\n";
        std::cout << "Base LR        : " << learningRate << "\n";
        std::cout << "Loss           : MC integral vs SH colour per splat\n";
    }

    // Group samples by splat index
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

    for (const auto& [splatIdx, sampleList] : groups) {
        if (sampleList.empty()) continue;

        DisneyBRDFParamsSimple& p = params[splatIdx];
        AdamStateAD& ad = adam[splatIdx];

        float prevLoss = 1e10f;
        int   stagnantCount = 0;

        for (int iter = 0; iter < maxIterations; ++iter) {

            // One gradient call per iteration -- the function integrates
            // over all samples internally (the MC estimate of the integral).
            BRDFGradients g = computeGradientAD(p, sampleList);

            if (g.loss == 0.f) break; // no valid samples

            // Gradient clipping
            glm::vec3 grad_bc = glm::clamp(g.bc, glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
            float     grad_met = glm::clamp(g.metallic, -GRAD_CLIP, GRAD_CLIP);
            float     grad_rough = glm::clamp(g.roughness, -GRAD_CLIP, GRAD_CLIP);
            float     grad_spec = glm::clamp(g.specular, -GRAD_CLIP, GRAD_CLIP);

            // Adam update
            ad.t++;
            p.baseColor -= ad.stepVec3(grad_bc, learningRate * LR_BC);
            p.metallic -= ad.stepScalar(grad_met, ad.m_met, ad.v_met, learningRate * LR_MET);
            p.roughness -= ad.stepScalar(grad_rough, ad.m_rough, ad.v_rough, learningRate * LR_ROUGH);
            p.specular -= ad.stepScalar(grad_spec, ad.m_spec, ad.v_spec, learningRate * LR_SPEC);

            p.clamp();

            // Early stop: converged
            if (g.loss < 1e-6f) break;

            // Early stop: stagnation
            if (std::abs(prevLoss - g.loss) < 1e-8f) {
                if (++stagnantCount > 30) break;
            }
            else {
                stagnantCount = 0;
            }
            prevLoss = g.loss;
        }

        ++processed;
        if (verbose && processed % progressInterval == 0)
            std::cout << "Progress: " << (processed * 100 / (int)groups.size())
            << "%\r" << std::flush;
    }

    if (verbose) std::cout << "\nOptimization complete!\n";

    // Write CSV
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