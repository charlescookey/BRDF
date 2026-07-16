#pragma once

// TwoPassBRDFOptimizer.h
//
// Addresses the flat-Li problem in 3DGS-derived BRDF estimation.
//
// ROOT CAUSE
//   3DGS SH coefficients bake in tonemapped appearance, so Li values from
//   hemisphere ray-marching are bounded to [~0.15, ~0.89] with no dynamic
//   range. This forces baseColor → ~1 regardless of true material colour,
//   because the optimizer can only match Lo by saturating albedo.
//
// TWO-PASS STRATEGY
//
//   Pass 1  —  Standard BRDF fit on raw (Li, Lo) samples.
//              Recovers roughness / specular / metallic reasonably well
//              (they shape the angular variation of Lo across omega_o groups).
//              baseColor is unreliable (collapses toward 1).
//
//   Scale step  —  For each omega_o group, compute
//                    s(wo)  =  Lo_observed(wo) / pred_Lo_pass1(wo)
//                  applied per channel (r, g, b) to preserve colour cues.
//                  Because pred_Lo < Lo_observed (Li is too flat), s > 1 in
//                  most groups, pushing Li up.  The directional variation of s
//                  across groups is kept intact (Option 1) so Pass 3 sees both
//                  a better magnitude AND better angular Li variation.
//
//   Pass 3  —  Re-run BRDF fit with  Li_scaled = s(wo) * Li.
//              Li now has realistic dynamic range; baseColor can separate
//              from the Li scale and converge to a physically plausible value.
//
// USAGE
//   TwoPassResult r = optimizeTwoPass(samples, gaussians);
//   // r.pass1_brdf  — result before Li correction
//   // r.pass3_brdf  — result after  Li correction
//   // Results written to two_pass_brdf.csv

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <functional>

#include "Math.h"
#include "Sampling.h"
#include "BRDFSample.h"
#include "BRDF_Optim_AutoDiff.h"
#include "happly.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================
//  Shared omega_o grouping key  (same rounding as the existing optimizer)
// ============================================================
namespace TwoPass {

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

inline OoKey makeKey(const glm::vec3& wo) {
    return {
        (int)std::round(wo.x * 10000.f),
        (int)std::round(wo.y * 10000.f),
        (int)std::round(wo.z * 10000.f)
    };
}

} // namespace TwoPass


// ============================================================
//  Per-group scale record
// ============================================================

struct GroupScale {
    glm::vec3 omega_o;         // view direction for this group

    glm::vec3 Lo_obs;          // observed outgoing radiance
    glm::vec3 pred_Lo_pass1;   // BRDF-predicted Lo using raw Li

    // Scale factor applied to every Li sample in this group:
    //   s = Lo_obs / max(pred_Lo, eps)
    // Kept per-channel so colour cues survive the correction.
    glm::vec3 s;
};


// ============================================================
//  Inner Adam optimization loop
//  Same logic as optimizeDisneyBRDFAutodiff but returns the
//  fitted params instead of writing to CSV.
// ============================================================

static DisneyBRDFParamsSimple runOptimizerPass(
    const std::vector<const BRDFSample*>& samples,
    const std::string&                    passLabel,
    int                                   maxIterations = 5000,
    float                                 learningRate  = 0.01f,
    bool                                  verbose       = true,
    bool                                  pinMetallic   = false)  // true → metallic fixed at 0
{
    constexpr float LR_BC    = 0.3f;
    constexpr float LR_MET   = 0.1f;
    constexpr float LR_ROUGH = 0.1f;
    constexpr float LR_SPEC  = 0.1f;
    constexpr float GRAD_CLIP = 10.f;

    if (verbose)
        std::cout << "\n--- " << passLabel << " ---\n"
                  << "Samples: " << samples.size()
                  << "  Iterations: " << maxIterations
                  << (pinMetallic ? "  [metallic pinned=0]" : "") << "\n";

    // Random initialisation (same seed each pass for reproducibility)
    DisneyBRDFParamsSimple p;
    {
        MTRandom rng(42);
        p.baseColor = glm::vec3(rng.next(), rng.next(), rng.next());
        p.metallic  = 0.f;   // always start at 0; pinMetallic keeps it there
        p.roughness = 0.5f;
        p.specular  = 0.5f;
    }

    AdamStateAD adam;
    float prevLoss    = 1e10f;
    int   stagnant    = 0;

    for (int iter = 0; iter < maxIterations; ++iter) {
        BRDFGradients g = computeGradientAD(p, samples);

        if (g.loss == 0.f || g.loss < 1e-7f) break;

        glm::vec3 grad_bc    = glm::clamp(g.bc,         glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
        float     grad_met   = glm::clamp(g.metallic,   -GRAD_CLIP, GRAD_CLIP);
        float     grad_rough = glm::clamp(g.roughness,  -GRAD_CLIP, GRAD_CLIP);
        float     grad_spec  = glm::clamp(g.specular,   -GRAD_CLIP, GRAD_CLIP);

        adam.t++;
        p.baseColor -= adam.stepVec3(grad_bc, learningRate * LR_BC);
        if (!pinMetallic)
            p.metallic -= adam.stepScalar(grad_met, adam.m_met, adam.v_met, learningRate * LR_MET);
        p.roughness -= adam.stepScalar(grad_rough, adam.m_rough, adam.v_rough, learningRate * LR_ROUGH);
        p.specular  -= adam.stepScalar(grad_spec,  adam.m_spec,  adam.v_spec,  learningRate * LR_SPEC);
        p.clamp();

        if (std::abs(prevLoss - g.loss) < 1e-8f) {
            if (++stagnant > 50) break;
        } else {
            stagnant = 0;
        }
        prevLoss = g.loss;

        if (verbose && iter % 1000 == 0)
            std::cout << "  iter " << std::setw(5) << iter
                      << "  loss=" << std::setw(10) << g.loss
                      << "  bc=(" << p.baseColor.r << ","
                                  << p.baseColor.g << ","
                                  << p.baseColor.b << ")"
                      << "  rough=" << p.roughness
                      << "\n";
    }

    if (verbose)
        std::cout << passLabel << " final:"
                  << "  loss="  << prevLoss
                  << "  bc=("   << p.baseColor.r << ","
                                << p.baseColor.g << ","
                                << p.baseColor.b << ")"
                  << "  rough=" << p.roughness
                  << "  spec="  << p.specular
                  << "  met="   << p.metallic  << "\n";

    return p;
}


// ============================================================
//  Scale step
//  Evaluate the Pass-1 BRDF on each omega_o group and compute
//  per-channel s = Lo_obs / pred_Lo.
// ============================================================

std::vector<GroupScale> computeGroupScales(
    const DisneyBRDFParamsSimple&         brdf,
    const std::vector<const BRDFSample*>& samples,
    bool                                  verbose = true)
{
    using namespace TwoPass;

    // Group samples by omega_o
    std::unordered_map<OoKey, std::vector<const BRDFSample*>, OoHash> byOo;
    for (const BRDFSample* s : samples)
        byOo[makeKey(s->omega_o)].push_back(s);

    std::vector<GroupScale> result;
    result.reserve(byOo.size());

    constexpr float EPS = 1e-4f;    // prevents division by zero
    constexpr float S_MAX = 50.f;   // prevents runaway scale on near-zero pred_Lo

    for (auto& [key, group] : byOo) {
        float sum_fr_li_r = 0.f, sum_fr_li_g = 0.f, sum_fr_li_b = 0.f;
        float lo_r = 0.f, lo_g = 0.f, lo_b = 0.f;
        int   validCount = 0;

        for (const BRDFSample* s : group) {
            float len = glm::length(s->omega_i);
            if (len < 1e-6f) continue;
            glm::vec3 wi = s->omega_i / len;
            if (glm::dot(s->normal, wi) <= 0.f) continue;

            float li_r = glm::max(0.f, s->L_i.r);
            float li_g = glm::max(0.f, s->L_i.g);
            float li_b = glm::max(0.f, s->L_i.b);

            // Evaluate BRDF as plain float — no autodiff needed here
            float fr_r, fr_g, fr_b;
            DisneyAD::evaluate<float>(
                brdf.baseColor.r, brdf.baseColor.g, brdf.baseColor.b,
                brdf.metallic, brdf.roughness, brdf.specular,
                s->omega_o, wi, s->normal,
                fr_r, fr_g, fr_b);

            sum_fr_li_r += fr_r * li_r;
            sum_fr_li_g += fr_g * li_g;
            sum_fr_li_b += fr_b * li_b;

            lo_r = s->L_o.r;
            lo_g = s->L_o.g;
            lo_b = s->L_o.b;
            ++validCount;
        }

        if (validCount == 0) continue;

        double n = (double)validCount;
        float pred_r = (float)(sum_fr_li_r * M_PI / n);
        float pred_g = (float)(sum_fr_li_g * M_PI / n);
        float pred_b = (float)(sum_fr_li_b * M_PI / n);

        GroupScale gs;
        gs.omega_o       = group[0]->omega_o;
        gs.Lo_obs        = glm::vec3(lo_r, lo_g, lo_b);
        gs.pred_Lo_pass1 = glm::vec3(pred_r, pred_g, pred_b);

        // Per-channel scale: how much must Li grow to close the gap?
        gs.s.r = glm::clamp(lo_r / glm::max(pred_r, EPS), 0.01f, S_MAX);
        gs.s.g = glm::clamp(lo_g / glm::max(pred_g, EPS), 0.01f, S_MAX);
        gs.s.b = glm::clamp(lo_b / glm::max(pred_b, EPS), 0.01f, S_MAX);

        result.push_back(gs);
    }

    if (verbose) {
        glm::vec3 meanS(0.f);
        for (auto& gs : result) meanS += gs.s;
        if (!result.empty()) meanS /= (float)result.size();

        std::cout << "\n--- Scale step ---\n"
                  << "Groups: " << result.size() << "\n"
                  << "Mean s  (r,g,b): ("
                  << meanS.r << ", " << meanS.g << ", " << meanS.b << ")\n";

        // Per-group breakdown
        for (auto& gs : result) {
            std::cout << "  wo=(" << std::setw(7) << std::fixed << std::setprecision(3)
                      << gs.omega_o.x << "," << gs.omega_o.y << "," << gs.omega_o.z << ")"
                      << "  Lo_obs=(" << gs.Lo_obs.r << "," << gs.Lo_obs.g << "," << gs.Lo_obs.b << ")"
                      << "  pred=("   << gs.pred_Lo_pass1.r << "," << gs.pred_Lo_pass1.g << "," << gs.pred_Lo_pass1.b << ")"
                      << "  s=("      << gs.s.r << "," << gs.s.g << "," << gs.s.b << ")\n";
        }
    }

    return result;
}


// ============================================================
//  Apply per-group scales to Li
//  Returns a new sample list — original samples are unchanged.
// ============================================================

std::vector<BRDFSample> applyGroupScales(
    const std::vector<BRDFSample>&  samples,
    const std::vector<GroupScale>&  scales)
{
    using namespace TwoPass;

    // Build omega_o -> scale lookup
    std::unordered_map<OoKey, glm::vec3, OoHash> lookup;
    for (const GroupScale& gs : scales)
        lookup[makeKey(gs.omega_o)] = gs.s;

    std::vector<BRDFSample> scaled = samples;   // copy all fields

    int tagged = 0;
    for (BRDFSample& s : scaled) {
        auto it = lookup.find(makeKey(s.omega_o));
        if (it == lookup.end()) continue;

        glm::vec3 sv = it->second;
        s.L_i.r = glm::max(0.f, s.L_i.r) * sv.r;
        s.L_i.g = glm::max(0.f, s.L_i.g) * sv.g;
        s.L_i.b = glm::max(0.f, s.L_i.b) * sv.b;
        ++tagged;
    }

    std::cout << "Applied group scales to " << tagged << " / "
              << samples.size() << " samples.\n";

    // Quick stats on new Li range
    float li_min = FLT_MAX, li_max = -FLT_MAX, li_sum = 0.f;
    for (const BRDFSample& s : scaled) {
        float lum = (s.L_i.r + s.L_i.g + s.L_i.b) / 3.f;
        li_min = std::min(li_min, lum);
        li_max = std::max(li_max, lum);
        li_sum += lum;
    }
    if (!scaled.empty())
        std::cout << "Scaled Li luminance  min=" << li_min
                  << "  max=" << li_max
                  << "  mean=" << li_sum / (float)scaled.size() << "\n";

    return scaled;
}



// ============================================================
//  Global Li scale from Pass-1 BRDF
//
//  For each omega_o group, computes:
//      s(wo) = Lo_obs(wo) / pred_Lo_pass1(wo)
//  then returns the value at `brightPercentile` (default 0.75).
//
//  Using the 75th-percentile instead of the mean avoids letting
//  very dark groups (Lo ≈ 0) drag the global scale back toward 1.
//  The selected s represents "how much brighter Li needs to be
//  in the better-lit directions to explain Lo_obs."
// ============================================================

float computeGlobalLiScale(
    const DisneyBRDFParamsSimple& brdf,
    const std::vector<const BRDFSample*>& samples,
    float                                 brightPercentile = 0.75f,
    bool                                  verbose = true)
{
    using namespace TwoPass;
    DisneyBRDFParamsSimple p = brdf;

    std::unordered_map<OoKey, std::vector<const BRDFSample*>, OoHash> byOo;
    for (const BRDFSample* s : samples)
        byOo[makeKey(s->omega_o)].push_back(s);

    constexpr float EPS = 1e-4f;
    std::vector<float> groupScaleLums;
    groupScaleLums.reserve(byOo.size());

    for (auto& [key, group] : byOo) {
        float sum_fr_li = 0.f;
        glm::vec3 lo_obs(0.f);
        int valid = 0;

        for (const BRDFSample* s : group) {
            float len = glm::length(s->omega_i);
            if (len < 1e-6f) continue;
            glm::vec3 wi = s->omega_i / len;
            if (glm::dot(s->normal, wi) <= 0.f) continue;

            float fr_r, fr_g, fr_b;
            DisneyAD::evaluate<float>(
                p.baseColor.r, p.baseColor.g, p.baseColor.b,
                p.metallic, p.roughness, p.specular,
                s->omega_o, wi, s->normal,
                fr_r, fr_g, fr_b);

            float li_lum = (glm::max(0.f, s->L_i.r)
                + glm::max(0.f, s->L_i.g)
                + glm::max(0.f, s->L_i.b)) / 3.f;
            float fr_lum = (fr_r + fr_g + fr_b) / 3.f;
            sum_fr_li += fr_lum * li_lum;
            lo_obs = s->L_o;   // all samples in group share omega_o => same Lo_obs
            ++valid;
        }
        if (valid == 0) continue;

        float pred_lo = (float)(sum_fr_li * M_PI / (double)valid);
        float lo_lum = (lo_obs.r + lo_obs.g + lo_obs.b) / 3.f;
        float s_lum = lo_lum / glm::max(pred_lo, EPS);
        groupScaleLums.push_back(s_lum);
    }

    if (groupScaleLums.empty()) return 1.f;

    std::sort(groupScaleLums.begin(), groupScaleLums.end());
    int idx = std::min(
        (int)(brightPercentile * (float)groupScaleLums.size()),
        (int)groupScaleLums.size() - 1);
    float globalScale = groupScaleLums[idx];

    if (verbose) {
        std::cout << "\n--- Global Li scale (p" << (int)(brightPercentile * 100) << ") ---\n"
            << "Groups : " << groupScaleLums.size() << "\n"
            << "Min s  : " << groupScaleLums.front() << "\n"
            << "Max s  : " << groupScaleLums.back() << "\n"
            << "Chosen : " << globalScale << "\n";
    }

    return globalScale;
}


// ============================================================
//  Save scaled Gaussians to a new PLY file
//
//  Reads every vertex property from `inputPlyPath`, replaces the
//  SH fields (f_dc_0/1/2, f_rest_0..44) with scaled values, and
//  writes a new binary PLY to `outputPlyPath`.
//
//  Scaling formula:
//    displayed colour = SH_C0 * ZeroSH + 0.5
//    new_f_dc   = scale * f_dc + (scale - 1) * 0.5 / SH_C0
//    new_f_rest = scale * f_rest        (no offset, zero-mean bands)
//
//  All other fields (position, opacity, rotation, Gaussian scale)
//  are copied unchanged.
// ============================================================

void saveScaledPLY(
    const std::string& inputPlyPath,
    const std::string& outputPlyPath,
    float              scale,
    bool               verbose = true)
{
    constexpr float SH_C0_LOCAL = 0.28209479177387814f;

    if (verbose)
        std::cout << "\n--- Saving scaled PLY ---\n"
        << "Input  : " << inputPlyPath << "\n"
        << "Output : " << outputPlyPath << "\n"
        << "Scale  : " << scale << "\n";

    happly::PLYData plyIn(inputPlyPath);
    auto& verts = plyIn.getElement("vertex");
    size_t N = verts.count;

    // ---- Read every field ----
    auto x = verts.getProperty<float>("x");
    auto y = verts.getProperty<float>("y");
    auto z = verts.getProperty<float>("z");

    auto dc0 = verts.getProperty<float>("f_dc_0");
    auto dc1 = verts.getProperty<float>("f_dc_1");
    auto dc2 = verts.getProperty<float>("f_dc_2");

    // 45 higher-order SH bands (3 channels × 15 coefficients)
    std::vector<std::vector<float>> rest(45);
    for (int b = 0; b < 45; ++b)
        rest[b] = verts.getProperty<float>("f_rest_" + std::to_string(b));

    auto sc0 = verts.getProperty<float>("scale_0");
    auto sc1 = verts.getProperty<float>("scale_1");
    auto sc2 = verts.getProperty<float>("scale_2");

    auto op = verts.getProperty<float>("opacity");

    auto r0 = verts.getProperty<float>("rot_0");
    auto r1 = verts.getProperty<float>("rot_1");
    auto r2 = verts.getProperty<float>("rot_2");
    auto r3 = verts.getProperty<float>("rot_3");

    // ---- Apply scale ----
    const float offset_correction = (scale - 1.f) * 0.5f / SH_C0_LOCAL;

    for (size_t i = 0; i < N; ++i) {
        dc0[i] = scale * dc0[i] + offset_correction;
        dc1[i] = scale * dc1[i] + offset_correction;
        dc2[i] = scale * dc2[i] + offset_correction;
    }
    for (int b = 0; b < 45; ++b)
        for (size_t i = 0; i < N; ++i)
            rest[b][i] *= scale;

    // ---- Write output ----
    happly::PLYData plyOut;
    plyOut.addElement("vertex", N);
    auto& ov = plyOut.getElement("vertex");

    ov.addProperty<float>("x", x);
    ov.addProperty<float>("y", y);
    ov.addProperty<float>("z", z);
    ov.addProperty<float>("f_dc_0", dc0);
    ov.addProperty<float>("f_dc_1", dc1);
    ov.addProperty<float>("f_dc_2", dc2);
    for (int b = 0; b < 45; ++b)
        ov.addProperty<float>("f_rest_" + std::to_string(b), rest[b]);
    ov.addProperty<float>("opacity", op);
    ov.addProperty<float>("scale_0", sc0);
    ov.addProperty<float>("scale_1", sc1);
    ov.addProperty<float>("scale_2", sc2);
    ov.addProperty<float>("rot_0", r0);
    ov.addProperty<float>("rot_1", r1);
    ov.addProperty<float>("rot_2", r2);
    ov.addProperty<float>("rot_3", r3);

    plyOut.write(outputPlyPath, happly::DataFormat::Binary);

    if (verbose)
        std::cout << "Saved " << N << " Gaussians → " << outputPlyPath << "\n";
};



// ============================================================
//  Top-level two-pass optimizer
// ============================================================

struct TwoPassResult {
    DisneyBRDFParamsSimple pass1_brdf;
    DisneyBRDFParamsSimple pass3_brdf;
    std::vector<GroupScale> groupScales;
    std::vector<BRDFSample> scaledSamples;   // Li-corrected samples used for Pass 3
    float globalLiScale = 1.f;              // p75 group scale used to produce the adjusted PLY
};

// Callback type: given the path to the adjusted PLY, load it, rebuild the
// Gaussian BVH, re-shoot Li rays, and return fresh BRDFSamples.
// The caller supplies this because it owns the camera / canvas / mesh BVH.
using ResampleFn = std::function<std::vector<BRDFSample>(const std::string& adjustedPlyPath)>;

TwoPassResult optimizeTwoPass(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>&         gaussians,
    int                            maxIterations   = 5000,
    const std::string&             inputPlyPath    = "",
    const std::string&             outputPlyPath   = "",
    float                          learningRate    = 0.01f,
    bool                           verbose         = true,
    const std::string&             csvPath         = "two_pass_brdf.csv",
    float                          scalePercentile = 0.90f,
    ResampleFn                     resampleFn      = nullptr)
{
    TwoPassResult result;

    if (samples.empty()) {
        std::cerr << "[TwoPass] No samples provided.\n";
        return result;
    }

    auto makePtrs = [](const std::vector<BRDFSample>& v) {
        std::vector<const BRDFSample*> ptrs(v.size());
        for (size_t i = 0; i < v.size(); ++i) ptrs[i] = &v[i];
        return ptrs;
    };

    // === Pass 1: raw Li — metallic pinned to 0 ===
    // Prevents the degenerate metallic=1 solution that collapses bc→1
    // and makes the scale step a no-op.
    auto ptrs1 = makePtrs(samples);
    result.pass1_brdf = runOptimizerPass(
        ptrs1, "Pass 1  (raw Li, metallic=0)", maxIterations, learningRate, verbose,
        /*pinMetallic=*/ true);

    // === Scale step: compute global Li uplift ===
    result.groupScales   = computeGroupScales(result.pass1_brdf, ptrs1, verbose);
    result.globalLiScale = computeGlobalLiScale(
        result.pass1_brdf, ptrs1, scalePercentile, verbose);

    // === Save adjusted PLY ===
    if (!inputPlyPath.empty() && !outputPlyPath.empty()) {
        saveScaledPLY(inputPlyPath, outputPlyPath, result.globalLiScale, verbose);
    } else if (verbose) {
        std::cout << "[TwoPass] globalLiScale=" << result.globalLiScale
                  << "  (no PLY paths supplied — skipping saveScaledPLY)\n";
    }

    // === Acquire Pass-3 samples ===
    // If a resample callback is provided, load the adjusted PLY, rebuild the
    // Gaussian scene, and re-shoot Li rays so Pass 3 sees real HDR radiance.
    // Otherwise fall back to the in-memory per-group scale (weaker but still
    // useful when the caller cannot re-sample).
    if (resampleFn && !outputPlyPath.empty()) {
        if (verbose)
            std::cout << "\n[TwoPass] Re-sampling Li from adjusted PLY: "
                      << outputPlyPath << "\n";
        result.scaledSamples = resampleFn(outputPlyPath);
        if (verbose)
            std::cout << "[TwoPass] Got " << result.scaledSamples.size()
                      << " fresh samples for Pass 3.\n";
    } else {
        if (verbose)
            std::cout << "\n[TwoPass] No resampleFn — using in-memory per-group scale.\n";
        result.scaledSamples = applyGroupScales(samples, result.groupScales);
    }

    // === Pass 3: re-sampled (or scaled) Li — metallic free ===
    auto ptrs3 = makePtrs(result.scaledSamples);
    result.pass3_brdf = runOptimizerPass(
        ptrs3, "Pass 3  (adjusted Li)", maxIterations, learningRate, verbose,
        /*pinMetallic=*/ false);

    // === Summary comparison ===
    if (verbose) {
        const auto& p1 = result.pass1_brdf;
        const auto& p3 = result.pass3_brdf;
        std::cout << "\n=== Two-Pass Summary ===\n"
                  << std::fixed << std::setprecision(4)
                  << "           baseColor              rough   spec   met\n"
                  << "Pass 1  :  ("
                  << p1.baseColor.r << ", " << p1.baseColor.g << ", " << p1.baseColor.b << ")  "
                  << p1.roughness << "  " << p1.specular << "  " << p1.metallic << "\n"
                  << "Pass 3  :  ("
                  << p3.baseColor.r << ", " << p3.baseColor.g << ", " << p3.baseColor.b << ")  "
                  << p3.roughness << "  " << p3.specular << "  " << p3.metallic << "\n";
    }

    // === CSV export ===
    std::ofstream csv(csvPath);
    if (csv.is_open()) {
        csv << "pass,"
               "baseColor.r,baseColor.g,baseColor.b,"
               "metallic,roughness,specular\n";

        auto writeRow = [&](const std::string& label, const DisneyBRDFParamsSimple& p) {
            csv << label << ","
                << p.baseColor.r << "," << p.baseColor.g << "," << p.baseColor.b << ","
                << p.metallic    << ","
                << p.roughness   << ","
                << p.specular    << "\n";
        };
        writeRow("pass1", result.pass1_brdf);
        writeRow("pass3", result.pass3_brdf);

        // Also log group scales
        csv << "\nomega_o.x,omega_o.y,omega_o.z,"
               "Lo_obs.r,Lo_obs.g,Lo_obs.b,"
               "pred_Lo.r,pred_Lo.g,pred_Lo.b,"
               "s.r,s.g,s.b\n";
        for (const GroupScale& gs : result.groupScales) {
            csv << gs.omega_o.x << "," << gs.omega_o.y << "," << gs.omega_o.z << ","
                << gs.Lo_obs.r  << "," << gs.Lo_obs.g  << "," << gs.Lo_obs.b  << ","
                << gs.pred_Lo_pass1.r << "," << gs.pred_Lo_pass1.g << "," << gs.pred_Lo_pass1.b << ","
                << gs.s.r << "," << gs.s.g << "," << gs.s.b << "\n";
        }

        // Global Li scale row
        csv << "\nglobal_li_scale,percentile\n"
            << result.globalLiScale << "," << scalePercentile << "\n";

        csv.close();
        if (verbose)
            std::cout << "Results written to " << csvPath << "\n";
    }

    return result;
}

