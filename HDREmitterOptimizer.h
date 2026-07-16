#pragma once

// HDREmitterOptimizer.h
//
// Joint Disney BRDF + per-emitter-Gaussian HDR-scale optimizer.
//
// EMITTER CLASSIFICATION
//   Uses scale (exp of the stored log-scale) and view-independent brightness,
//   NOT mesh distance, because the SuGAR OBJ may include background/cloud
//   geometry that would place sky Gaussians close to the mesh surface.
//
//   Criterion:  max_exp_scale > scaleThreshold
//            && view_independent_brightness >= brightnessPercentile
//
//   Background/sky Gaussians in 3DGS are characteristically large (high scale)
//   because they represent wide-area diffuse regions.
//
// JOINT OPTIMIZATION  (coordinate-descent Adam)
//   Step A — hold emitter log-scales fixed as floats, update BRDF params.
//   Step B — hold BRDF params fixed as floats, update each emitter log-scale
//             with autodiff (one var per emitter, single backward pass per emitter).
//
//   Emitter scale is parameterised as  scale = exp(log_scale),
//   initialised to 0.0 (→ scale = 1.0, no change from pre-trained SH colour).
//   log_scale is clamped to [0, 4] so scale stays in [1, ~55].
//
// VALIDATION RENDER
//   renderEmittersOnly() builds a sub-BVH from just the classified Gaussians
//   and renders the scene with them, letting you inspect what was selected.

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include "Math.h"
#include "mesh.h"
#include "BRDFSample.h"
#include "BRDF_Optim_AutoDiff.h"
#include "utilities.h"
#include "Sampling.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================
//  1.  Emitter classification
// ============================================================

struct EmitterSet
{
    std::unordered_set<int> indices;          // Gaussian indices classified as emitters
    float scaleThreshold    = 0.f;            // threshold used during classification
    float brightnessThreshold = 0.f;          // percentile brightness used

    bool isEmitter(int idx) const { return indices.count(idx) > 0; }
};

// Classify Gaussians as emitters by:
//   1. max exp(scale) > scaleThreshold       — large splats are background / sky
//   2. view-independent brightness >= brightnessPercentile — only keep the bright ones
//
// scaleThreshold:      try 0.3–1.0 (scene units after exp()).
// brightnessPercentile: 0.85 means top 15% by brightness among large Gaussians.
EmitterSet classifyEmitters(
    const std::vector<Gaussian>& gaussians,
    float scaleThreshold     = 0.5f,
    float brightnessPercentile = 0.85f)
{
    EmitterSet result;
    result.scaleThreshold = scaleThreshold;

    // --- Step 1: filter by Gaussian scale ---
    std::vector<int> largeIndices;
    largeIndices.reserve(gaussians.size() / 4);

    for (int i = 0; i < (int)gaussians.size(); ++i) {
        const Gaussian& g = gaussians[i];
        // scale field stores log-scales; exp gives the actual Gaussian radius.
        float maxScale = std::exp(g.scale._max());
        if (maxScale > scaleThreshold)
            largeIndices.push_back(i);
    }

    std::cout << "[HDREmitter] " << largeIndices.size() << " / " << gaussians.size()
              << " Gaussians passed the scale filter (max_exp_scale > "
              << scaleThreshold << ").\n";

    if (largeIndices.empty()) {
        std::cerr << "[HDREmitter] Warning: no Gaussians passed the scale filter. "
                     "Try a smaller scaleThreshold.\n";
        return result;
    }

    // --- Step 2: filter by view-independent brightness ---
    // viewIndependent() = SH_C0 * ZeroSH + 0.5
    std::vector<float> brightnesses(largeIndices.size());
    for (int j = 0; j < (int)largeIndices.size(); ++j) {
        const Gaussian& g = gaussians[largeIndices[j]];
        Vec3 c = g.ZeroSH * SH_C0 + Vec3(0.5f);
        brightnesses[j] = (c.x + c.y + c.z) / 3.f;
    }

    std::vector<float> sorted = brightnesses;
    std::sort(sorted.begin(), sorted.end());
    int cutoff = std::min((int)(brightnessPercentile * (float)sorted.size()),
                          (int)sorted.size() - 1);
    float brightnessThresh = sorted[cutoff];
    result.brightnessThreshold = brightnessThresh;

    std::cout << "[HDREmitter] Brightness threshold (p"
              << (int)(brightnessPercentile * 100.f) << "): "
              << brightnessThresh << "\n";

    for (int j = 0; j < (int)largeIndices.size(); ++j) {
        if (brightnesses[j] >= brightnessThresh)
            result.indices.insert(largeIndices[j]);
    }

    std::cout << "[HDREmitter] " << result.indices.size()
              << " emitter Gaussians selected.\n";

    return result;
}


// ============================================================
//  2.  Validation render — scene with only emitter Gaussians
// ============================================================

// Renders the scene using only the Gaussians in emitters.
// Builds a temporary sub-BVH, renders into canvas, saves to outPNG.
// Use this to visually verify the classification before running the optimizer.
void renderEmittersOnly(
    const EmitterSet&           emitters,
    std::vector<Gaussian>&      allGaussians,
    Camera&                     camera,
    GamesEngineeringBase::Window* canvas,
    const std::string&          outPNG = "emitters_only.png")
{
    // Extract subset
    std::vector<Gaussian> subset;
    subset.reserve(emitters.indices.size());
    for (auto& g : allGaussians)
        if (emitters.isEmitter(g.index))
            subset.push_back(g);

    std::cout << "[HDREmitter] Rendering " << subset.size()
              << " emitter Gaussians...\n";

    if (subset.empty()) {
        std::cerr << "[HDREmitter] No emitters to render.\n";
        return;
    }

    // Build temporary BVH
    BVHNode emitterBVH;
    emitterBVH.build(subset);

    // Clear canvas to black before rendering
    int W = (int)camera.width, H = (int)camera.height;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            canvas->draw(x, y, 0.f, 0.f, 0.f);

    renderImageSH(camera, canvas, subset, &emitterBVH);
    savePNG(outPNG, canvas);

    std::cout << "[HDREmitter] Saved emitter-only render to " << outPNG << "\n";
}


// ============================================================
//  3.  Per-sample emitter annotation
// ============================================================

// Records which emitter Gaussian dominated a Li sample and its base (pre-scale) Li.
// dominant_emitter_idx == -1  =>  this Li sample is not emitter-tagged.
struct EmitterTag
{
    int       dominant_emitter_idx = -1;
    glm::vec3 base_Li              = glm::vec3(0.f);
    bool      escaped_mesh         = false;   // true if Li ray did not hit any mesh
};


// ============================================================
//  4.  GaussianColor with dominant-emitter tracking
// ============================================================

// Like GaussianColor() in utilities.h but also returns the index of the
// Gaussian with the highest (alpha * transmittance) weight, IF it is
// a member of the emitter set.
static Colour GaussianColorWithEmitter(
    Ray&                    ray,
    std::vector<Gaussian>&  gaussians,
    const EmitterSet&       emitters,
    int&                    dominantEmitterIdx,
    int&                    contribution_count)
{
    dominantEmitterIdx = -1;
    contribution_count = 0;

    struct Hit { float t; Gaussian* g; };
    std::vector<Hit> hits;
    hits.reserve(gaussians.size());

    for (auto& g : gaussians) {
        float t = ray.dir.dot((g.pos - ray.o));
        if (t <= 0.f) continue;
        hits.push_back({ t, &g });
    }
    std::sort(hits.begin(), hits.end(),
              [](const Hit& a, const Hit& b) { return a.t < b.t; });

    Colour color(0, 0, 0);
    float  tr     = 1.f;
    float  bestW  = 0.f;

    for (auto& h : hits) {
        Gaussian& g = *h.g;
        float alpha = g.computeAlpha(ray);
        if (tr < 0.001f) break;

        float w = alpha * tr;
        Vec3   viewDir = (ray.o - g.pos).normalize();
        Colour SHColor = evaluateSphericalHarmonics(viewDir, g);

        if (w > 0.05f) {
            color = color + (SHColor * alpha * tr);
            tr   *= (1.f - alpha);
            ++contribution_count;

            if (emitters.isEmitter(g.index) && w > bestW) {
                bestW              = w;
                dominantEmitterIdx = g.index;
            }
        }
    }
    return color;
}


// ============================================================
//  5.  Emitter-tagged Li hemisphere sampling
//      Drop-in replacement for monteCarloSamplingHit.
//      Outputs a BRDFSample and an EmitterTag for every valid hemisphere sample.
// ============================================================

void monteCarloSamplingHitTagged(
    MTRandom&               sampler,
    MeshHit&                hit,
    std::vector<Gaussian>&  all,
    BVHNode*                gaussBVH,
    const MeshBVH&          meshBVH,      // SuGAR mesh BVH for escape test
    const EmitterSet&       emitters,
    const glm::vec3&        omega_o,
    std::vector<BRDFSample>& outSamples,
    std::vector<EmitterTag>& outTags,
    float                   contribution,
    int                     threadID = 0,
    int                     N_SAMPLES = 20)
{
    Vec3 normal = hit.normal;
    if (normal.dot(fromGLM(omega_o)) < 0.f)
        normal = normal * -1.f;

    Frame frame;
    frame.fromVector(normal);

    for (int s = 0; s < N_SAMPLES; ++s) {
        Vec3 localDir      = SamplingDistributions::cosineSampleHemisphere(
                                 sampler.next(), sampler.next());
        Vec3 omega_i_world = frame.toWorld(localDir);

        float NdotL = normal.dot(omega_i_world);
        if (NdotL <= 0.f) continue;

        // Build Li ray
        Ray liRay;
        liRay.init(hit.hitPoint + (omega_i_world * EPSILON), omega_i_world);

        // --- Mesh escape test ---
        // A ray that hits the mesh (any part, including clouds) is interreflecting
        // off a real surface, not arriving from an open emitter.
        MeshHit meshCheck  = meshBVH.traverse(liRay);
        bool    escapedMesh = !meshCheck.hit;

        // --- Gaussian compositing (with emitter tracking) ---
        // Use a dedicated BVH slot so the Lo slot isn't clobbered.
        gaussBVH->traverse(liRay, all, threadID + threadNum * 2);
        int  contribution_count = 0;
        int  dominantEmitter    = -1;

        Colour LiColor = GaussianColorWithEmitter(
            liRay,
            gaussBVH->getIntersectedGaussiansVec(threadID + threadNum * 2),
            emitters,
            dominantEmitter,
            contribution_count);

        // Only promote to emitter if the ray actually escaped — hitting the mesh
        // means the colour comes from reflected light, not a true emitter.
        if (!escapedMesh)
            dominantEmitter = -1;

        glm::vec3 L_i = glm::clamp(LiColor.ToGlm(), glm::vec3(0.f), glm::vec3(10.f));

        // --- Build BRDFSample ---
        BRDFSample sample{};
        sample.splatIndex = 0;
        sample.omega_i    = omega_i_world.ToGlm();
        sample.omega_o    = omega_o;
        sample.normal     = normal.ToGlm();
        sample.L_i        = L_i;
        sample.L_o        = glm::vec3(0.f);   // filled by caller
        sample.cosTheta   = NdotL;
        sample.weight     = contribution;

        // --- Emitter tag ---
        EmitterTag tag{};
        tag.dominant_emitter_idx = dominantEmitter;
        tag.base_Li              = L_i;    // unscaled SH Li, stored for optimizer
        tag.escaped_mesh         = escapedMesh;

        outSamples.push_back(sample);
        outTags.push_back(tag);
    }
}


// ============================================================
//  6.  Adam helper for emitter log-scale scalars
// ============================================================

struct AdamStateEmitter
{
    float m = 0.f, v = 0.f;
    int   t = 0;

    static constexpr float beta1   = 0.9f;
    static constexpr float beta2   = 0.999f;
    static constexpr float eps     = 1e-8f;

    float step(float grad, float lr)
    {
        ++t;
        m = beta1 * m + (1.f - beta1) * grad;
        v = beta2 * v + (1.f - beta2) * grad * grad;
        float mh = m / (1.f - std::pow(beta1, (float)t));
        float vh = v / (1.f - std::pow(beta2, (float)t));
        return lr * mh / (std::sqrt(vh) + eps);
    }
};


// ============================================================
//  7.  Gradient helpers
// ============================================================

// --- Step A: BRDF gradient with emitter scales applied as constant floats ---
static BRDFGradients computeGradientWithEmitters(
    const DisneyBRDFParamsSimple&              p,
    const std::vector<const BRDFSample*>&      samples,
    const std::vector<const EmitterTag*>&      tags,
    const std::unordered_map<int, float>&      emitter_log_scales)
{
    var bc_r(p.baseColor.r), bc_g(p.baseColor.g), bc_b(p.baseColor.b);
    var met(p.metallic), rough(p.roughness), spec(p.specular);

    // Group by omega_o (same keying as existing optimizer)
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

    using Group = std::pair<const BRDFSample*, const EmitterTag*>;
    std::unordered_map<OoKey, std::vector<Group>, OoHash> byOo;

    for (int i = 0; i < (int)samples.size(); ++i) {
        const BRDFSample* s = samples[i];
        OoKey key {
            (int)std::round(s->omega_o.x * 10000.f),
            (int)std::round(s->omega_o.y * 10000.f),
            (int)std::round(s->omega_o.z * 10000.f)
        };
        byOo[key].push_back({ s, tags[i] });
    }

    var totalLoss(0.0);
    int groupsUsed = 0;

    for (auto& [key, group] : byOo) {
        var  sum_r(0.0), sum_g(0.0), sum_b(0.0);
        int  validCount = 0;
        float lo_r = 0.f, lo_g = 0.f, lo_b = 0.f;

        for (auto& [s, tag] : group) {
            float len = glm::length(s->omega_i);
            if (len < 1e-6f) continue;
            glm::vec3 wi = s->omega_i / len;
            if (glm::dot(s->normal, wi) <= 0.f) continue;

            // Apply emitter scale as a constant float — BRDF sees scaled Li
            glm::vec3 base_Li = (tag && tag->dominant_emitter_idx >= 0)
                                ? tag->base_Li : s->L_i;
            float scale = 1.f;
            if (tag && tag->dominant_emitter_idx >= 0) {
                auto it = emitter_log_scales.find(tag->dominant_emitter_idx);
                if (it != emitter_log_scales.end())
                    scale = std::exp(it->second);
            }

            float li_r = glm::max(0.f, base_Li.r * scale);
            float li_g = glm::max(0.f, base_Li.g * scale);
            float li_b = glm::max(0.f, base_Li.b * scale);

            var fr, fg, fb;
            DisneyAD::evaluate(bc_r, bc_g, bc_b, met, rough, spec,
                               s->omega_o, wi, s->normal, fr, fg, fb);

            sum_r += fr * val(li_r);
            sum_g += fg * val(li_g);
            sum_b += fb * val(li_b);

            lo_r = s->L_o.r; lo_g = s->L_o.g; lo_b = s->L_o.b;
            ++validCount;
        }
        if (validCount == 0) continue;

        double n = (double)validCount;
        var pred_r = sum_r * (M_PI / n);
        var pred_g = sum_g * (M_PI / n);
        var pred_b = sum_b * (M_PI / n);
        var res_r  = pred_r - val(lo_r);
        var res_g  = pred_g - val(lo_g);
        var res_b  = pred_b - val(lo_b);

        totalLoss += res_r * res_r + res_g * res_g + res_b * res_b;
        ++groupsUsed;
    }

    if (groupsUsed == 0)
        return BRDFGradients{ glm::vec3(0.f), 0.f, 0.f, 0.f, 0.f };

    totalLoss = totalLoss / val((double)groupsUsed);

    auto [d_bc_r, d_bc_g, d_bc_b, d_met, d_rough, d_spec] =
        derivatives(totalLoss, wrt(bc_r, bc_g, bc_b, met, rough, spec));

    BRDFGradients out;
    out.bc       = glm::vec3((float)d_bc_r, (float)d_bc_g, (float)d_bc_b);
    out.metallic  = (float)d_met;
    out.roughness = (float)d_rough;
    out.specular  = (float)d_spec;
    out.loss      = (float)val(totalLoss);
    return out;
}


// --- Step B: gradient of loss w.r.t. one emitter's log_scale (BRDF fixed) ---
static float computeEmitterLogScaleGradient(
    const DisneyBRDFParamsSimple&         p,
    const std::vector<const BRDFSample*>& samples,
    const std::vector<const EmitterTag*>& tags,
    int                                   emitter_idx,
    float                                 current_log_scale)
{
    var log_sc(current_log_scale);

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

    using Group = std::pair<const BRDFSample*, const EmitterTag*>;
    std::unordered_map<OoKey, std::vector<Group>, OoHash> byOo;

    for (int i = 0; i < (int)samples.size(); ++i) {
        const BRDFSample* s = samples[i];
        OoKey key {
            (int)std::round(s->omega_o.x * 10000.f),
            (int)std::round(s->omega_o.y * 10000.f),
            (int)std::round(s->omega_o.z * 10000.f)
        };
        byOo[key].push_back({ s, tags[i] });
    }

    var totalLoss(0.0);
    int groupsUsed = 0;

    for (auto& [key, group] : byOo) {
        var  sum_r(0.0), sum_g(0.0), sum_b(0.0);
        int  validCount = 0;
        float lo_r = 0.f, lo_g = 0.f, lo_b = 0.f;

        for (auto& [s, tag] : group) {
            float len = glm::length(s->omega_i);
            if (len < 1e-6f) continue;
            glm::vec3 wi = s->omega_i / len;
            if (glm::dot(s->normal, wi) <= 0.f) continue;

            // BRDF evaluated as plain float (BRDF params are held fixed in Step B)
            float fr_r, fr_g, fr_b;
            DisneyAD::evaluate<float>(
                p.baseColor.r, p.baseColor.g, p.baseColor.b,
                p.metallic, p.roughness, p.specular,
                s->omega_o, wi, s->normal,
                fr_r, fr_g, fr_b);

            // Li is differentiable only for samples whose dominant emitter == emitter_idx
            var eff_li_r, eff_li_g, eff_li_b;
            if (tag && tag->dominant_emitter_idx == emitter_idx) {
                var scale  = exp(log_sc);   // autodiff exp, grad flows through log_sc
                eff_li_r = val(glm::max(0.f, tag->base_Li.r)) * scale;
                eff_li_g = val(glm::max(0.f, tag->base_Li.g)) * scale;
                eff_li_b = val(glm::max(0.f, tag->base_Li.b)) * scale;
            } else {
                // Constant w.r.t. log_sc — no gradient contribution
                eff_li_r = var(glm::max(0.f, s->L_i.r));
                eff_li_g = var(glm::max(0.f, s->L_i.g));
                eff_li_b = var(glm::max(0.f, s->L_i.b));
            }

            sum_r += fr_r * eff_li_r;
            sum_g += fr_g * eff_li_g;
            sum_b += fr_b * eff_li_b;

            lo_r = s->L_o.r; lo_g = s->L_o.g; lo_b = s->L_o.b;
            ++validCount;
        }
        if (validCount == 0) continue;

        double n = (double)validCount;
        var pred_r = sum_r * (M_PI / n);
        var pred_g = sum_g * (M_PI / n);
        var pred_b = sum_b * (M_PI / n);
        var res_r  = pred_r - val(lo_r);
        var res_g  = pred_g - val(lo_g);
        var res_b  = pred_b - val(lo_b);

        totalLoss += res_r * res_r + res_g * res_g + res_b * res_b;
        ++groupsUsed;
    }

    if (groupsUsed == 0) return 0.f;

    totalLoss = totalLoss / val((double)groupsUsed);
    auto [d_log] = derivatives(totalLoss, wrt(log_sc));
    return (float)d_log;
}


// ============================================================
//  8.  Joint optimizer
// ============================================================

struct JointOptResult {
    DisneyBRDFParamsSimple          brdf;
    std::unordered_map<int, float>  emitter_log_scales;   // gaussian_idx -> log_scale
    std::unordered_map<int, float>  emitter_scales;       // gaussian_idx -> exp(log_scale)
    float                           finalLoss = 0.f;
};

JointOptResult optimizeJointBRDFAndEmitters(
    const std::vector<BRDFSample>& samples,
    const std::vector<EmitterTag>& tags,
    std::vector<Gaussian>&         gaussians,
    const EmitterSet&              emitters,
    int                            maxIterations  = 2000,
    float                          lr_brdf        = 0.01f,
    float                          lr_emitter     = 0.02f,
    bool                           verbose        = true)
{
    constexpr float LR_BC    = 0.3f;
    constexpr float LR_MET   = 0.1f;
    constexpr float LR_ROUGH = 0.1f;
    constexpr float LR_SPEC  = 0.1f;
    constexpr float GRAD_CLIP = 10.f;

    if (verbose) {
        std::cout << "=== Joint Disney BRDF + HDR Emitter Optimizer ===\n";
        std::cout << "Samples    : " << samples.size() << "\n";
        std::cout << "Emitters   : " << emitters.indices.size() << "\n";
        std::cout << "Iterations : " << maxIterations << "\n";
    }

    // Collect which emitters actually appear in these samples
    std::unordered_set<int> activeEmitters;
    for (const auto& tag : tags)
        if (tag.dominant_emitter_idx >= 0)
            activeEmitters.insert(tag.dominant_emitter_idx);

    if (verbose)
        std::cout << "Active emitters in samples: " << activeEmitters.size() << "\n";

    // Initialise emitter log-scales to 0 (scale = 1)
    std::unordered_map<int, float>           emitter_log_scales;
    std::unordered_map<int, AdamStateEmitter> emitterAdam;
    for (int idx : activeEmitters) {
        emitter_log_scales[idx] = 0.f;
        emitterAdam[idx]        = AdamStateEmitter{};
    }

    // Initialise BRDF
    DisneyBRDFParamsSimple p;
    {
        MTRandom rng(42);
        p.baseColor = glm::vec3(rng.next(), rng.next(), rng.next());
        p.metallic  = 0.f;
        p.roughness = 0.5f;
        p.specular  = 0.5f;
    }
    AdamStateAD brdfAdam;

    // Build const pointer vectors (gradient functions take these)
    std::vector<const BRDFSample*> samplePtrs(samples.size());
    std::vector<const EmitterTag*> tagPtrs(tags.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        samplePtrs[i] = &samples[i];
        tagPtrs[i]    = &tags[i];
    }

    float lastLoss = 1e10f;

    for (int iter = 0; iter < maxIterations; ++iter) {

        // === Step A: update BRDF params (emitter scales are fixed floats) ===
        BRDFGradients g = computeGradientWithEmitters(
            p, samplePtrs, tagPtrs, emitter_log_scales);

        if (g.loss < 1e-7f) break;

        glm::vec3 grad_bc    = glm::clamp(g.bc, glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
        float     grad_met   = glm::clamp(g.metallic,  -GRAD_CLIP, GRAD_CLIP);
        float     grad_rough = glm::clamp(g.roughness, -GRAD_CLIP, GRAD_CLIP);
        float     grad_spec  = glm::clamp(g.specular,  -GRAD_CLIP, GRAD_CLIP);

        brdfAdam.t++;
        p.baseColor -= brdfAdam.stepVec3(grad_bc,    lr_brdf * LR_BC);
        p.metallic  -= brdfAdam.stepScalar(grad_met,   brdfAdam.m_met,   brdfAdam.v_met,   lr_brdf * LR_MET);
        p.roughness -= brdfAdam.stepScalar(grad_rough, brdfAdam.m_rough, brdfAdam.v_rough, lr_brdf * LR_ROUGH);
        p.specular  -= brdfAdam.stepScalar(grad_spec,  brdfAdam.m_spec,  brdfAdam.v_spec,  lr_brdf * LR_SPEC);
        p.clamp();

        // === Step B: update each emitter log-scale (BRDF is fixed) ===
        for (int eIdx : activeEmitters) {
            float d = computeEmitterLogScaleGradient(
                p, samplePtrs, tagPtrs, eIdx, emitter_log_scales[eIdx]);
            d = glm::clamp(d, -GRAD_CLIP, GRAD_CLIP);
            emitter_log_scales[eIdx] -= emitterAdam[eIdx].step(d, lr_emitter);
            // Keep log_scale in [0, 4]:  scale in [1, ~55x]
            // Floor at 0 prevents emitter scales from shrinking below 1 (SH baseline).
            emitter_log_scales[eIdx] = glm::clamp(emitter_log_scales[eIdx], 0.f, 4.f);
        }

        lastLoss = g.loss;

        if (verbose && iter % 200 == 0) {
            float minSc = FLT_MAX, maxSc = -FLT_MAX;
            for (auto& [i, ls] : emitter_log_scales) {
                float s = std::exp(ls);
                minSc = std::min(minSc, s);
                maxSc = std::max(maxSc, s);
            }
            std::cout << "Iter " << std::setw(4) << iter
                      << "  loss="  << g.loss
                      << "  bc=("   << p.baseColor.r << ","
                                    << p.baseColor.g << ","
                                    << p.baseColor.b << ")"
                      << "  rough=" << p.roughness
                      << "  emitter_scale=["
                      << (activeEmitters.empty() ? 1.f : minSc) << ", "
                      << (activeEmitters.empty() ? 1.f : maxSc) << "]\n";
        }
    }

    JointOptResult result;
    result.brdf             = p;
    result.emitter_log_scales = emitter_log_scales;
    result.finalLoss        = lastLoss;
    for (auto& [idx, ls] : emitter_log_scales)
        result.emitter_scales[idx] = std::exp(ls);

    if (verbose) {
        std::cout << "=== Joint optimization complete ===\n";
        std::cout << "Final loss : " << lastLoss              << "\n";
        std::cout << "baseColor  : (" << p.baseColor.r << ", "
                                      << p.baseColor.g << ", "
                                      << p.baseColor.b << ")\n";
        std::cout << "roughness  : " << p.roughness           << "\n";
        std::cout << "specular   : " << p.specular            << "\n";
        std::cout << "metallic   : " << p.metallic            << "\n";

        if (!result.emitter_scales.empty()) {
            float sum = 0.f;
            for (auto& [i, s] : result.emitter_scales) sum += s;
            float mean = sum / (float)result.emitter_scales.size();
            std::cout << "Mean emitter scale: " << mean << "x\n";
        }
    }

    return result;
}
