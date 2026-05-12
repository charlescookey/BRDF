#pragma once
#pragma once

// Disney BRDF Optimizer autodiff edition
//
// Loss formulation:
//
//   Progressive per-sample SGD:
//
//     Samples are shuffled, then fed one at a time to the optimizer.
//     For each single sample (wi, wo, Li, Lo, N):
//
//       pred = BRDF(wi, wo, N) * Li * pi
//       loss = || pred - Lo ||^2
//
//   After all samples have been seen once (one epoch), a final
//   polish pass runs over all samples grouped by omega_o using the
//   full MC integral estimator:
//
//       pred(wo) = (pi / N_wo) * sum_{wi in group}[ BRDF(wi, wo, N) * Li ]
//       loss     = (1/G) * sum_wo || pred(wo) - Lo(wo) ||^2
//
//   This two-phase approach lets baseColor converge in the first
//   ~50 samples (strong gradient, view-independent), then roughness
//   and specular refine as grazing-angle samples arrive, and the
//   final polish locks in the full-integral estimate.

#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

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
// computeGradientSingleSample
//
// Loss for one sample:
//   pred = BRDF(wi, wo, N) * Li * pi
//   loss = || pred - Lo ||^2
//
// NdotL cancels with cosine-sampling PDF so it does NOT appear here.
// -------------------------------------------------------------------------
BRDFGradients computeGradientSingleSample_progressive(
    const DisneyBRDFParamsSimple& p,
    const BRDFSample* s
) {
    float len = glm::length(s->omega_i);
    if (len < 1e-6f)
        return BRDFGradients{ glm::vec3(0.f), 0.f, 0.f, 0.f, 0.f };

    glm::vec3 wi = s->omega_i / len;
    float     ndl = glm::dot(s->normal, wi);
    if (ndl <= 0.f)
        return BRDFGradients{ glm::vec3(0.f), 0.f, 0.f, 0.f, 0.f };

    var bc_r(p.baseColor.r), bc_g(p.baseColor.g), bc_b(p.baseColor.b);
    var met(p.metallic), rough(p.roughness), spec(p.specular);

    var fr, fg, fb;
    DisneyAD::evaluate(bc_r, bc_g, bc_b, met, rough, spec,
        s->omega_o, wi, s->normal,
        fr, fg, fb);

    // pred = BRDF * Li * pi  (NdotL cancels with cosine PDF)
    float li_r = glm::clamp(s->L_i.r, 0.f, 10.f);
    float li_g = glm::clamp(s->L_i.g, 0.f, 10.f);
    float li_b = glm::clamp(s->L_i.b, 0.f, 10.f);

    var pred_r = fr * val(li_r * M_PI);
    var pred_g = fg * val(li_g * M_PI);
    var pred_b = fb * val(li_b * M_PI);

    var res_r = pred_r - val(s->L_o.r);
    var res_g = pred_g - val(s->L_o.g);
    var res_b = pred_b - val(s->L_o.b);

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


// -------------------------------------------------------------------------
// computeGradientAD  (full MC integral, used for the polish pass)
//
// Groups samples by omega_o, computes one MC integral per group:
//   pred(wo) = (pi / N_wo) * sum_{wi}[ BRDF(wi,wo,N) * Li ]
//   loss     = (1/G) * sum_wo || pred(wo) - Lo(wo) ||^2
// -------------------------------------------------------------------------
BRDFGradients computeGradientAD_progressive(
    const DisneyBRDFParamsSimple& p,
    const std::vector<const BRDFSample*>& samples
) {
    var bc_r(p.baseColor.r), bc_g(p.baseColor.g), bc_b(p.baseColor.b);
    var met(p.metallic), rough(p.roughness), spec(p.specular);

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
        var   sum_r(0.0), sum_g(0.0), sum_b(0.0);
        int   validCount = 0;
        float lo_r = 0.f, lo_g = 0.f, lo_b = 0.f;

        for (const BRDFSample* s : group) {
            float len = glm::length(s->omega_i);
            if (len < 1e-6f) continue;
            glm::vec3 wi = s->omega_i / len;
            float     ndl = glm::dot(s->normal, wi);
            if (ndl <= 0.f) continue;

            float li_r = glm::clamp(s->L_i.r, 0.f, 10.f);
            float li_g = glm::clamp(s->L_i.g, 0.f, 10.f);
            float li_b = glm::clamp(s->L_i.b, 0.f, 10.f);

            var fr, fg, fb;
            DisneyAD::evaluate(bc_r, bc_g, bc_b, met, rough, spec,
                s->omega_o, wi, s->normal,
                fr, fg, fb);

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
// applyAdamStep  (shared helper)
// -------------------------------------------------------------------------
static void applyAdamStep(
    DisneyBRDFParamsSimple& p,
    AdamStateAD& ad,
    const BRDFGradients& g,
    float                   lr,
    float LR_BC, float LR_MET, float LR_ROUGH, float LR_SPEC,
    float GRAD_CLIP)
{
    glm::vec3 grad_bc = glm::clamp(g.bc, glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
    float     grad_met = glm::clamp(g.metallic, -GRAD_CLIP, GRAD_CLIP);
    float     grad_rough = glm::clamp(g.roughness, -GRAD_CLIP, GRAD_CLIP);
    float     grad_spec = glm::clamp(g.specular, -GRAD_CLIP, GRAD_CLIP);

    ad.t++;
    p.baseColor -= ad.stepVec3(grad_bc, lr * LR_BC);
    p.metallic -= ad.stepScalar(grad_met, ad.m_met, ad.v_met, lr * LR_MET);
    p.roughness -= ad.stepScalar(grad_rough, ad.m_rough, ad.v_rough, lr * LR_ROUGH);
    p.specular -= ad.stepScalar(grad_spec, ad.m_spec, ad.v_spec, lr * LR_SPEC);
    p.clamp();
}


// -------------------------------------------------------------------------
// optimizeDisneyBRDFAutodiff
//
// Phase 1 — Progressive SGD:
//   Samples for each splat are shuffled then fed one at a time.
//   STEPS_PER_SAMPLE Adam steps are taken per sample so early samples
//   (which are cheap and establish baseColor) contribute more than late
//   ones (which refine roughness/specular).
//
// Phase 2 — Full-batch polish:
//   After all samples have been seen, POLISH_ITERATIONS steps of the
//   grouped MC-integral loss refine the final estimate.
// -------------------------------------------------------------------------
void optimizeDisneyBRDFAutodiff_progressive(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>& gaussians,
    int   maxIterations = 500,   // polish iterations (phase 2)
    float learningRate = 0.01f,
    bool  verbose = true
) {
    constexpr float LR_BC = 0.3f;
    constexpr float LR_MET = 0.1f;
    constexpr float LR_ROUGH = 0.1f;
    constexpr float LR_SPEC = 0.1f;
    constexpr float GRAD_CLIP = 10.0f;

    // How many Adam steps to take per individual sample in phase 1.
    // 5-10 is a good balance: enough to move meaningfully, cheap enough
    // that 2048 samples * 10 steps = 20k evals, fast per splat.
    constexpr int   STEPS_PER_SAMPLE = 5;

    // Polish iterations after SGD phase (phase 2, full MC integral).
    const     int   POLISH_ITERS = maxIterations;

    if (verbose) {
        std::cout << "=== Disney BRDF Optimizer (progressive SGD + polish) ===\n";
        std::cout << "Steps per sample : " << STEPS_PER_SAMPLE << "\n";
        std::cout << "Polish iters     : " << POLISH_ITERS << "\n";
        std::cout << "Base LR          : " << learningRate << "\n";
    }

    // Group samples by splat
    std::unordered_map<int, std::vector<const BRDFSample*>> groups;
    for (const auto& s : samples)
        groups[s.splatIndex].push_back(&s);

    size_t maxIndex = 0;
    for (const auto& [idx, _] : groups)
        maxIndex = std::max<size_t>(maxIndex, (size_t)idx);

    // Initialise parameters from SH DC colour (good baseColor prior)
    std::vector<DisneyBRDFParamsSimple> params(maxIndex + 1);
    for (size_t i = 0; i < params.size() && i < gaussians.size(); ++i) {
        params[i].baseColor = glm::clamp(gaussians[i].testColor, 0.02f, 0.98f);
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

        // ------------------------------------------------------------------
        // Phase 1: progressive per-sample SGD
        //
        // Shuffle so we see a mix of view directions from the very first
        // sample, giving roughness/specular a gradient signal early on
        // rather than waiting until all near-normal samples are exhausted.
        // ------------------------------------------------------------------
        std::vector<const BRDFSample*> shuffled = sampleList;
        std::mt19937 rng(static_cast<unsigned>(splatIdx));
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        for (const BRDFSample* s : shuffled) {
            for (int step = 0; step < STEPS_PER_SAMPLE; ++step) {
                BRDFGradients g = computeGradientSingleSample_progressive(p, s);
                if (g.loss < 1e-8f) break;
                applyAdamStep(p, ad, g, learningRate,
                    LR_BC, LR_MET, LR_ROUGH, LR_SPEC, GRAD_CLIP);
            }
        }

        // ------------------------------------------------------------------
        // Phase 2: full-batch MC-integral polish
        //
        // Now that parameters are close to the true values, the grouped
        // MC estimator gives a low-variance loss signal to lock them in.
        // ------------------------------------------------------------------
        float prevLoss = 1e10f;
        int   stagnantCount = 0;

        for (int iter = 0; iter < POLISH_ITERS; ++iter) {
            BRDFGradients g = computeGradientAD_progressive(p, sampleList);
            if (g.loss < 1e-6f) break;

            applyAdamStep(p, ad, g, learningRate,
                LR_BC, LR_MET, LR_ROUGH, LR_SPEC, GRAD_CLIP);

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

    // Write results
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
