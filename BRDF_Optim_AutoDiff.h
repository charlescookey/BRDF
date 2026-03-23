#pragma once

// Disney BRDF Optimizer autodiff edition
//
// Loss formulation:
//
//   Samples are grouped by splat, then by omega_o within each splat.
//   For each (splat, omega_o) group we compute one MC integral:
//
//     pred(wo) = (pi / N_wo) * sum_{wi in group}[ BRDF(wi, wo, N) * Li ]
//
//   NdotL cancels with the cosine-sampling PDF (PDF = NdotL/pi), so it
//   does NOT appear inside the sum.
//
//   The loss is the sum of per-group squared residuals:
//
//     loss = sum_wo || pred(wo) - Lo(wo) ||^2
//
//   Each omega_o group contributes its own Lo (the SH colour evaluated in
//   that view direction). This preserves the view-dependent signal that
//   roughness and specular depend on -- mixing all omega_o into one sum
//   destroys that signal.

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "Math.h"
#include "BRDFSample.h"

#include <glm/glm.hpp>
#include <autodiff/reverse/var.hpp>

using namespace autodiff;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// -------------------------------------------------------------------------
// Adam optimizer state
// -------------------------------------------------------------------------
struct AdamStateAD {
    glm::vec3 m_bc = glm::vec3(0.f);
    float     m_met = 0.f, m_rough = 0.f, m_spec = 0.f;
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


// -------------------------------------------------------------------------
// Templated Disney BRDF
// -------------------------------------------------------------------------
namespace DisneyAD {

    template<typename T>
    inline T schlickFresnel(T u) {
        T m = T(1.0) - u;
        if (m < T(0.0)) m = T(0.0);
        if (m > T(1.0)) m = T(1.0);
        return m * m * m * m * m;
    }

    template<typename T>
    inline T GTR2(T NdotH, T roughness) {
        T a = roughness * roughness;
        T a2 = a * a;
        T t = T(1.0) + (a2 - T(1.0)) * NdotH * NdotH;
        return a2 / (T(M_PI) * t * t);
    }

    template<typename T>
    inline T smithG_GGX(T NdotV, T roughness) {
        T a = roughness * roughness;
        T a2 = a * a;
        T b = NdotV * NdotV;
        return T(1.0) / (NdotV + sqrt(a2 + b - a2 * b));
    }

    template<typename T>
    inline T disneyDiffuse(T NdotL, T NdotV, T LdotH, T roughness) {
        T fd90 = T(0.5) + T(2.0) * LdotH * LdotH * roughness;
        T FL = schlickFresnel(NdotL);
        T FV = schlickFresnel(NdotV);
        return (T(1.0) + (fd90 - T(1.0)) * FL) * (T(1.0) + (fd90 - T(1.0)) * FV);
    }

    template<typename T>
    inline void evaluate(
        T bc_r, T bc_g, T bc_b,
        T metallic, T roughness, T specular,
        const glm::vec3& V,
        const glm::vec3& L,
        const glm::vec3& N,
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

        T Cspec_r = (T(1.0) - metallic) * specular * T(0.08) + metallic * bc_r;
        T Cspec_g = (T(1.0) - metallic) * specular * T(0.08) + metallic * bc_g;
        T Cspec_b = (T(1.0) - metallic) * specular * T(0.08) + metallic * bc_b;

        T Fd = disneyDiffuse(NdotL, NdotV, LdotH, roughness);
        T invPI = T(1.0 / M_PI);
        T diff_r = bc_r * invPI * Fd;
        T diff_g = bc_g * invPI * Fd;
        T diff_b = bc_b * invPI * Fd;

        T D = GTR2(NdotH, roughness);
        T FH = schlickFresnel(LdotH);
        T F_r = Cspec_r + (T(1.0) - Cspec_r) * FH;
        T F_g = Cspec_g + (T(1.0) - Cspec_g) * FH;
        T F_b = Cspec_b + (T(1.0) - Cspec_b) * FH;
        T G = smithG_GGX(NdotL, roughness) * smithG_GGX(NdotV, roughness);

        out_r = (T(1.0) - metallic) * diff_r + G * F_r * D;
        out_g = (T(1.0) - metallic) * diff_g + G * F_g * D;
        out_b = (T(1.0) - metallic) * diff_b + G * F_b * D;
    }

} // namespace DisneyAD


// -------------------------------------------------------------------------
// BRDFGradients
// -------------------------------------------------------------------------
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
// Groups samples by omega_o, computes one MC integral per group, sums losses.
//
// For cosine-weighted sampling (PDF = NdotL/pi):
//   pred(wo) = (pi / N_wo) * sum_{wi}[ BRDF(wi,wo,N) * Li ]
//   loss     = (1/G) * sum_wo || pred(wo) - Lo(wo) ||^2
//
// where G = number of omega_o groups (so loss scale is independent of
// how many view directions were sampled).
// -------------------------------------------------------------------------
BRDFGradients computeGradientAD(
    const DisneyBRDFParamsSimple& p,
    const std::vector<const BRDFSample*>& samples
) {
    var bc_r(p.baseColor.r), bc_g(p.baseColor.g), bc_b(p.baseColor.b);
    var met(p.metallic), rough(p.roughness), spec(p.specular);

    // Group samples by omega_o.
    // Samples collected by collectSamplesForView share the same exact
    // omega_o float value, so rounding to 4dp is a safe key.
    struct OoKey {
        int x, y, z;
        bool operator==(const OoKey& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct OoHash {
        size_t operator()(const OoKey& k) const {
            size_t h = std::hash<int>{}(k.x);
            h ^= std::hash<int>{}(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>{}(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<OoKey, std::vector<const BRDFSample*>, OoHash> byOo;
    for (const BRDFSample* s : samples) {
        OoKey key{
            (int)std::round(s->omega_o.x * 10000.f),
            (int)std::round(s->omega_o.y * 10000.f),
            (int)std::round(s->omega_o.z * 10000.f)
        };
        byOo[key].push_back(s);
    }

    var totalLoss(0.0);
    int groupsUsed = 0;

    for (auto& [key, group] : byOo) {
        var sum_r(0.0), sum_g(0.0), sum_b(0.0);
        int validCount = 0;
        float lo_r = 0.f, lo_g = 0.f, lo_b = 0.f;

        for (const BRDFSample* s : group) {
            float len = glm::length(s->omega_i);
            if (len < 1e-6f) continue;
            glm::vec3 wi = s->omega_i / len;

            float ndl = glm::dot(s->normal, wi);
            if (ndl <= 0.f) continue;

            float li_r = glm::clamp(s->L_i.r, 0.f, 2.f);
            float li_g = glm::clamp(s->L_i.g, 0.f, 2.f);
            float li_b = glm::clamp(s->L_i.b, 0.f, 2.f);

            var fr, fg, fb;
            DisneyAD::evaluate(bc_r, bc_g, bc_b, met, rough, spec,
                s->omega_o, wi, s->normal,
                fr, fg, fb);

            // NdotL cancels with cosine-sampling PDF -- do NOT include it
            sum_r += fr * val(li_r);
            sum_g += fg * val(li_g);
            sum_b += fb * val(li_b);

            lo_r = s->L_o.r;
            lo_g = s->L_o.g;
            lo_b = s->L_o.b;

            ++validCount;
        }

        if (validCount == 0) continue;

        double n = static_cast<double>(validCount);
        var pred_r = sum_r * (M_PI / n);
        var pred_g = sum_g * (M_PI / n);
        var pred_b = sum_b * (M_PI / n);

        var res_r = pred_r - val(lo_r);
        var res_g = pred_g - val(lo_g);
        var res_b = pred_b - val(lo_b);

        totalLoss += res_r * res_r + res_g * res_g + res_b * res_b;
        ++groupsUsed;
    }

    if (groupsUsed == 0)
        return BRDFGradients{ glm::vec3(0.f), 0.f, 0.f, 0.f, 0.f };

    // Average over view groups
    totalLoss = totalLoss / val((double)groupsUsed);

    auto [d_bc_r, d_bc_g, d_bc_b, d_met, d_rough, d_spec] =
        derivatives(totalLoss, wrt(bc_r, bc_g, bc_b, met, rough, spec));

    BRDFGradients out;
    out.bc.r = static_cast<float>(d_bc_r);
    out.bc.g = static_cast<float>(d_bc_g);
    out.bc.b = static_cast<float>(d_bc_b);
    out.metallic = static_cast<float>(d_met);
    out.roughness = static_cast<float>(d_rough);
    out.specular = static_cast<float>(d_spec);
    out.loss = static_cast<float>(val(totalLoss));
    return out;
}


// -------------------------------------------------------------------------
// optimizeDisneyBRDFAutodiff
// -------------------------------------------------------------------------
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
        std::cout << "Loss           : per-omega_o MC integral vs SH colour\n";
    }

    std::unordered_map<int, std::vector<const BRDFSample*>> groups;
    for (const auto& s : samples)
        groups[s.splatIndex].push_back(&s);

    size_t maxIndex = 0;
    for (const auto& [idx, _] : groups)
        maxIndex = std::max<size_t>(maxIndex, (size_t)idx);

    std::vector<DisneyBRDFParamsSimple> params(maxIndex + 1);

    for (size_t i = 0; i < params.size() && i < gaussians.size(); ++i) {
        //params[i].baseColor = glm::clamp(gaussians[i].testColor, 0.02f, 0.98f);
        MTRandom sample(4);
        params[i].baseColor = glm::vec3(sample.next(), sample.next(), sample.next());
        params[i].metallic = 0.f;
        params[i].roughness = 0.5f;
        params[i].specular = 0.5f;
    }
    for (size_t i = gaussians.size(); i < params.size(); ++i)
        params[i].baseColor = glm::vec3(0.5f);

    if (verbose)
        std::cout << "Optimizing " << groups.size() << " splats, "
        << samples.size() << " total samples\n";

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
            BRDFGradients g = computeGradientAD(p, sampleList);

            if (g.loss == 0.f) break;

            glm::vec3 grad_bc = glm::clamp(g.bc, glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
            float     grad_met = glm::clamp(g.metallic, -GRAD_CLIP, GRAD_CLIP);
            float     grad_rough = glm::clamp(g.roughness, -GRAD_CLIP, GRAD_CLIP);
            float     grad_spec = glm::clamp(g.specular, -GRAD_CLIP, GRAD_CLIP);

            ad.t++;
            p.baseColor -= ad.stepVec3(grad_bc, learningRate * LR_BC);
            p.metallic -= ad.stepScalar(grad_met, ad.m_met, ad.v_met, learningRate * LR_MET);
            p.roughness -= ad.stepScalar(grad_rough, ad.m_rough, ad.v_rough, learningRate * LR_ROUGH);
            p.specular -= ad.stepScalar(grad_spec, ad.m_spec, ad.v_spec, learningRate * LR_SPEC);

            p.clamp();

            if (g.loss < 1e-6f) break;

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