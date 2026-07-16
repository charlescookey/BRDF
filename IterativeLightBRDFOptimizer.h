#pragma once

// IterativeLightBRDFOptimizer.h
//
// Simultaneously optimizes the Disney BRDF and the radiance scale of the
// Gaussian scene through alternating passes until both converge.
//
// MOTIVATION
//   3DGS SH coefficients bake tonemapped appearance, so Li is flat
//   (~0.15–0.89).  A single BRDF fit forces albedo → 1 to compensate.
//   Correcting Li once (TwoPassBRDFOptimizer) helps but is not enough
//   because the first correction is based on an overestimated albedo.
//   Iterating allows both unknowns to tighten together.
//
// ALGORITHM (alternating optimization)
//
//   samples_0  ←  original scene (flat Li)
//
//   k = 0
//   brdf_0  ←  optimizeBRDF(samples_0)   [metallic pinned=0 on first pass]
//
//   loop:
//       scale_k   =  p90( Lo_obs / predict(brdf_k, Li_k) )   per omega_o group
//       if |scale_k − 1| < ε  →  converged, stop
//       cumulativeScale  *=  scale_k
//       PLY_k+1   ←  saveScaledPLY(originalPLY, cumulativeScale)
//       samples_k+1  ←  resampleLi(PLY_k+1)    [Lo_obs kept from originals]
//       brdf_k+1  ←  optimizeBRDF(samples_k+1)
//       k++
//
//   Convergence: scale_k ≈ 1  (Li is already correct; BRDF no longer demands
//   more light to explain Lo).  Typically 3–6 iterations.
//   Divergence guard: if cumulativeScale grows past maxCumulativeScale, stop.
//
// USAGE
//   auto resample = [&](const std::string& plyPath) -> std::vector<BRDFSample> {
//       // load plyPath, rebuild BVH, re-shoot Li rays, keep original Lo_obs
//       ...
//   };
//
//   IterativeOptResult r = optimizeIterative(
//       samples, gaussians,
//       "train.ply", "train_adjusted.ply",
//       resample);
//
//   // r.finalBRDF        — converged BRDF parameters
//   // r.cumulativeScale  — total SH scale applied to the PLY
//   // r.passes           — per-iteration log
//   // Results written to iterative_brdf.csv

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "Math.h"
#include "BRDFSample.h"
#include "TwoPassBRDFOptimizer.h"   // runOptimizerPass, computeGlobalLiScale, saveScaledPLY

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// ============================================================
//  Per-iteration record
// ============================================================

struct IterPassResult {
    int   iteration;
    DisneyBRDFParamsSimple brdf;
    float scaleApplied;       // scale computed THIS iteration
    float cumulativeScale;    // total scale on the PLY after this iteration
    float loss;               // final BRDF loss for this pass
};


// ============================================================
//  Full result
// ============================================================

struct IterativeOptResult {
    std::vector<IterPassResult> passes;
    DisneyBRDFParamsSimple      finalBRDF;
    float                       cumulativeScale = 1.f;
    bool                        converged       = false;
};


// ============================================================
//  Main iterative optimizer
//
//  Parameters
//  ----------
//  samples          Original (Li, Lo) samples from the unmodified scene.
//  gaussians        Gaussian splats (not modified in place; just for context).
//  inputPlyPath     Original PLY — never overwritten.
//  outputPlyPath    Adjusted PLY — overwritten each iteration with cumulative scale.
//  resampleFn       Caller-supplied callback: given the adjusted PLY path,
//                   loads it, rebuilds the BVH, re-shoots Li hemisphere rays
//                   while keeping the original Lo_obs, returns fresh samples.
//  maxPasses        Hard cap on iterations (default 10).
//  convergenceEps   Stop when |scale − 1| < this (default 0.05 = 5%).
//  maxCumulativeScale  Divergence guard — stop if total scale exceeds this.
//  scalePercentile  Which group-scale percentile to use as the global uplift.
//  maxIterations    Adam iterations per BRDF pass.
//  learningRate     Adam learning rate.
//  csvPath          Path for the per-iteration CSV log.
//  verbose          Print progress to stdout.
// ============================================================

IterativeOptResult optimizeIterative(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>&         gaussians,
    const std::string&             inputPlyPath,
    const std::string&             outputPlyPath,
    ResampleFn                     resampleFn,
    int                            maxPasses          = 10,
    float                          convergenceEps     = 0.05f,
    float                          maxCumulativeScale = 20.f,
    float                          scalePercentile    = 0.90f,
    int                            maxIterations      = 5000,
    float                          learningRate       = 0.01f,
    const std::string&             csvPath            = "iterative_brdf.csv",
    bool                           verbose            = true)
{
    IterativeOptResult result;

    if (samples.empty()) {
        std::cerr << "[Iterative] No samples provided.\n";
        return result;
    }

    auto makePtrs = [](const std::vector<BRDFSample>& v) {
        std::vector<const BRDFSample*> ptrs(v.size());
        for (size_t i = 0; i < v.size(); ++i) ptrs[i] = &v[i];
        return ptrs;
    };

    // Working sample set — starts as originals, updated each iteration
    std::vector<BRDFSample> currentSamples = samples;
    float cumulativeScale = 1.f;
    DisneyBRDFParamsSimple prevBRDF;

    // ---- CSV header ----
    std::ofstream csv(csvPath);
    if (csv.is_open()) {
        csv << "iteration,scaleApplied,cumulativeScale,loss,"
               "bc.r,bc.g,bc.b,metallic,roughness,specular,converged\n";
    }

    for (int iter = 0; iter <= maxPasses; ++iter) {

        bool firstPass = (iter == 0);

        if (verbose)
            std::cout << "\n========================================\n"
                      << "  Iteration " << iter
                      << (firstPass ? "  [initial — metallic pinned]" : "") << "\n"
                      << "========================================\n";

        // ---- BRDF step: fix Li, solve BRDF ----
        auto ptrs = makePtrs(currentSamples);
        DisneyBRDFParamsSimple brdf = runOptimizerPass(
            ptrs,
            "Iter " + std::to_string(iter) + " BRDF",
            maxIterations, learningRate, verbose,
            /*pinMetallic=*/ firstPass);   // pin only on first pass

        // ---- Compute loss for logging ----
        BRDFGradients g = computeGradientAD(brdf, ptrs);
        float loss = g.loss;

        // ---- Li scale step: fix BRDF, solve scale ----
        float scaleApplied = computeGlobalLiScale(
            brdf, ptrs, scalePercentile, verbose);

        // ---- Record ----
        IterPassResult record;
        record.iteration       = iter;
        record.brdf            = brdf;
        record.scaleApplied    = scaleApplied;
        record.cumulativeScale = cumulativeScale * scaleApplied;
        record.loss            = loss;
        result.passes.push_back(record);

        if (csv.is_open()) {
            csv << iter << ","
                << scaleApplied << ","
                << record.cumulativeScale << ","
                << loss << ","
                << brdf.baseColor.r << "," << brdf.baseColor.g << "," << brdf.baseColor.b << ","
                << brdf.metallic    << ","
                << brdf.roughness   << ","
                << brdf.specular    << ","
                << (std::abs(scaleApplied - 1.f) < convergenceEps ? "YES" : "no") << "\n";
            csv.flush();
        }

        if (verbose) {
            std::cout << "\n[Iter " << iter << "] "
                      << "scale=" << scaleApplied
                      << "  cumulative=" << record.cumulativeScale
                      << "  loss=" << loss
                      << "  bc=(" << brdf.baseColor.r << ","
                                  << brdf.baseColor.g << ","
                                  << brdf.baseColor.b << ")"
                      << "  rough=" << brdf.roughness
                      << "  met="   << brdf.metallic << "\n";
        }

        // ---- Convergence check ----
        if (std::abs(scaleApplied - 1.f) < convergenceEps) {
            std::cout << "\n[Iterative] Converged at iteration " << iter
                      << "  (|scale-1|=" << std::abs(scaleApplied - 1.f)
                      << " < eps=" << convergenceEps << ")\n";
            result.converged       = true;
            result.finalBRDF       = brdf;
            result.cumulativeScale = record.cumulativeScale;
            break;
        }

        // ---- Divergence guard ----
        if (record.cumulativeScale > maxCumulativeScale) {
            std::cerr << "\n[Iterative] WARNING: cumulativeScale="
                      << record.cumulativeScale
                      << " exceeded maxCumulativeScale=" << maxCumulativeScale
                      << " — stopping to avoid divergence.\n";
            result.finalBRDF       = brdf;
            result.cumulativeScale = record.cumulativeScale;
            break;
        }

        // ---- Last iteration: don't resample ----
        if (iter == maxPasses) {
            result.finalBRDF       = brdf;
            result.cumulativeScale = record.cumulativeScale;
            if (verbose)
                std::cout << "[Iterative] Reached maxPasses=" << maxPasses
                          << " without converging.\n";
            break;
        }

        // ---- Update cumulative scale & write adjusted PLY ----
        cumulativeScale *= scaleApplied;
        saveScaledPLY(inputPlyPath, outputPlyPath, cumulativeScale, verbose);

        // ---- Li step: fix BRDF, resample from brighter scene ----
        if (verbose)
            std::cout << "[Iterative] Re-sampling Li from: " << outputPlyPath << "\n";

        currentSamples = resampleFn(outputPlyPath);

        if (verbose)
            std::cout << "[Iterative] Got " << currentSamples.size()
                      << " fresh samples.\n";

        prevBRDF = brdf;
    }

    if (csv.is_open()) csv.close();

    if (verbose) {
        const auto& f = result.finalBRDF;
        std::cout << "\n=== Iterative Result ===\n"
                  << "Converged      : " << (result.converged ? "YES" : "NO") << "\n"
                  << "Passes run     : " << result.passes.size() << "\n"
                  << "CumulativeScale: " << result.cumulativeScale << "\n"
                  << "Final bc       : (" << f.baseColor.r << ", "
                                          << f.baseColor.g << ", "
                                          << f.baseColor.b << ")\n"
                  << "Final roughness: " << f.roughness << "\n"
                  << "Final specular : " << f.specular  << "\n"
                  << "Final metallic : " << f.metallic  << "\n"
                  << "CSV written to : " << csvPath     << "\n";
    }

    return result;
}
