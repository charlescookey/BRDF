#pragma once

// SpherePointOptim.h
// ------------------
// Optimizer for recovering Disney BRDF parameters from a SINGLE surface point
// on an analytic sphere scene.
//
// Pipeline:
//   1. Run SphereSampleGen.py  → sphere_samples.csv
//   2. Call loadSphereSamples()  to read the CSV into BRDFSample objects
//   3. Call optimizeSpherePoint() to recover base_color, metallic, roughness, specular
//
// The optimizer reuses computeGradientAD() from BRDF_Optim_AutoDiff.h verbatim.
// The loss formula is identical:
//
//   For each omega_o group:
//     pred(wo) = (pi / N_wo) * sum_wi [ BRDF(wi, wo, N) * Li(wi) ]
//     loss     = (1/G) * sum_wo || pred(wo) - Lo(wo) ||^2
//
// This guarantees the Python ground truth and the C++ optimizer speak the same
// radiometric language, so recovery errors are purely optimisation artefacts
// and not unit mismatches.

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

#include "Math.h"
#include "BRDFSample.h"
#include "DisneyBRDFOptimizerSimple.h"
#include "BRDF_Optim_AutoDiff.h"   // DisneyAD, AdamStateAD, BRDFGradients, computeGradientAD

#include <glm/glm.hpp>


// ─────────────────────────────────────────────────────────────────────────────
// loadSphereSamples
//
// Reads the CSV produced by SphereSampleGen.py.
// CSV columns:
//   omega_i.x/y/z, omega_o.x/y/z, normal.x/y/z, L_i.x/y/z, L_o.x/y/z
//
// All samples are assigned splatIndex = 0 (single point).
// shColour is set to L_o so the existing computeGradientAD grouping logic
// can retrieve Lo without modification.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<BRDFSample> loadSphereSamples(const std::string& filename)
{
    std::vector<BRDFSample> samples;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[SpherePointOptim] Cannot open '" << filename << "'\n"
                  << "  Did you run:  python SphereSampleGen.py  ?\n";
        return samples;
    }

    std::string line;
    std::getline(file, line);   // skip header row

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string tok;
        float v[15];
        int col = 0;
        while (col < 15 && std::getline(ss, tok, ',')) {
            try { v[col++] = std::stof(tok); }
            catch (...) { break; }
        }
        if (col < 15) continue;   // malformed row

        BRDFSample s{};
        s.splatIndex = 0;
        s.omega_i  = glm::vec3(v[0],  v[1],  v[2]);
        s.omega_o  = glm::vec3(v[3],  v[4],  v[5]);
        s.normal   = glm::vec3(v[6],  v[7],  v[8]);
        s.L_i      = glm::vec3(v[9],  v[10], v[11]);
        s.L_o      = glm::vec3(v[12], v[13], v[14]);
        s.cosTheta = glm::dot(s.normal, s.omega_i);
        s.weight   = 1.0f;
        s.shColour = s.L_o;   // mirrors L_o; computeGradientAD reads L_o directly

        samples.push_back(s);
    }

    std::cout << "[SpherePointOptim] Loaded " << samples.size()
              << " samples from '" << filename << "'\n";
    return samples;
}


// ─────────────────────────────────────────────────────────────────────────────
// optimizeSpherePoint
//
// Runs Adam gradient descent to recover Disney BRDF parameters for the single
// point whose samples were loaded with loadSphereSamples().
//
// Parameters:
//   samples    - output of loadSphereSamples()
//   maxIter    - maximum optimiser iterations (default 1000)
//   lr         - base Adam learning rate (default 0.01)
//   verbose    - print progress every 100 iterations
//
// The initial guess is deliberately neutral (grey, mid roughness) so that
// convergence demonstrates genuine information recovery, not a warm start.
// ─────────────────────────────────────────────────────────────────────────────
inline void optimizeSpherePoint(
    const std::vector<BRDFSample>& samples,
    int   maxIter = 1000,
    float lr      = 0.01f,
    bool  verbose = true)
{
    if (samples.empty()) {
        std::cerr << "[SpherePointOptim] No samples to optimize.\n";
        return;
    }

    // Build raw-pointer list expected by computeGradientAD
    std::vector<const BRDFSample*> ptrs;
    ptrs.reserve(samples.size());
    for (const auto& s : samples)
        ptrs.push_back(&s);

    // ── Initial guess (intentionally far from ground truth) ─────────────────
    DisneyBRDFParamsSimple p;
    p.baseColor = glm::vec3(0.5f, 0.5f, 0.5f);   // GT: (0.8, 0.2, 0.1)
    p.metallic  = 0.0f;                            // GT: 0.0
    p.roughness = 0.5f;                            // GT: 0.3
    p.specular  = 0.5f;                            // GT: 0.5

    AdamStateAD adam{};

    // Per-parameter learning rate multipliers (same as optimizeDisneyBRDFAutodiff)
    constexpr float LR_BC    = 0.3f;
    constexpr float LR_MET   = 0.1f;
    constexpr float LR_ROUGH = 0.1f;
    constexpr float LR_SPEC  = 0.1f;
    constexpr float GRAD_CLIP = 10.0f;

    if (verbose) {
        std::cout << "\n";
        std::cout << "==================================================\n";
        std::cout << "  Sphere Point BRDF Optimizer\n";
        std::cout << "==================================================\n";
        std::cout << "  Samples      : " << samples.size() << "\n";
        std::cout << "  Max iters    : " << maxIter << "\n";
        std::cout << "  Learning rate: " << lr << "\n";
        std::cout << "  Init  bc     = (0.5, 0.5, 0.5)\n";
        std::cout << "        met    = 0.0\n";
        std::cout << "        rough  = 0.5\n";
        std::cout << "        spec   = 0.5\n";
        std::cout << "--------------------------------------------------\n";
        std::cout << std::left
                  << std::setw(8)  << "Iter"
                  << std::setw(14) << "Loss"
                  << std::setw(24) << "BaseColor"
                  << std::setw(10) << "Metallic"
                  << std::setw(10) << "Roughness"
                  << std::setw(10) << "Specular"
                  << "\n";
        std::cout << "--------------------------------------------------\n";
    }

    float prevLoss = 1e10f;
    int   stagnant = 0;

    for (int iter = 0; iter < maxIter; ++iter) {

        BRDFGradients g = computeGradientAD(p, ptrs);

        if (g.loss == 0.f) break;

        // Gradient clipping
        glm::vec3 gbc  = glm::clamp(g.bc,        glm::vec3(-GRAD_CLIP), glm::vec3(GRAD_CLIP));
        float     gmet = glm::clamp(g.metallic,  -GRAD_CLIP, GRAD_CLIP);
        float     grgh = glm::clamp(g.roughness, -GRAD_CLIP, GRAD_CLIP);
        float     gspc = glm::clamp(g.specular,  -GRAD_CLIP, GRAD_CLIP);

        // Adam step
        adam.t++;
        p.baseColor -= adam.stepVec3(gbc, lr * LR_BC);
        p.metallic  -= adam.stepScalar(gmet, adam.m_met,   adam.v_met,   lr * LR_MET);
        p.roughness -= adam.stepScalar(grgh, adam.m_rough, adam.v_rough, lr * LR_ROUGH);
        p.specular  -= adam.stepScalar(gspc, adam.m_spec,  adam.v_spec,  lr * LR_SPEC);
        p.clamp();

        if (verbose && (iter % 100 == 0 || iter == maxIter - 1)) {
            std::cout << std::fixed
                      << std::left  << std::setw(8) << iter
                      << std::setw(14) << std::setprecision(6) << g.loss
                      << "("   << std::setprecision(3) << p.baseColor.r
                      << ", "  << p.baseColor.g
                      << ", "  << p.baseColor.b << ")  "
                      << std::setw(10) << p.metallic
                      << std::setw(10) << p.roughness
                      << std::setw(10) << p.specular
                      << "\n";
        }

        // Convergence checks
        if (g.loss < 1e-7f) {
            std::cout << "  [converged at iter " << iter << "]\n";
            break;
        }
        if (std::abs(prevLoss - g.loss) < 1e-9f) {
            if (++stagnant > 50) {
                std::cout << "  [stagnated at iter " << iter << "]\n";
                break;
            }
        } else {
            stagnant = 0;
        }
        prevLoss = g.loss;
    }

    // ── Final report ─────────────────────────────────────────────────────────
    std::cout << "\n";
    std::cout << "==================================================\n";
    std::cout << "  RECOVERED BRDF PARAMETERS\n";
    std::cout << "==================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  baseColor = ("
              << p.baseColor.r << ", "
              << p.baseColor.g << ", "
              << p.baseColor.b << ")\n";
    std::cout << "  metallic  = " << p.metallic  << "\n";
    std::cout << "  roughness = " << p.roughness << "\n";
    std::cout << "  specular  = " << p.specular  << "\n";
    std::cout << "\n";
    std::cout << "  GROUND TRUTH (from SphereSampleGen.py)\n";
    std::cout << "  baseColor = (0.8000, 0.2000, 0.1000)\n";
    std::cout << "  metallic  = 0.0000\n";
    std::cout << "  roughness = 0.3000\n";
    std::cout << "  specular  = 0.5000\n";
    std::cout << "==================================================\n";
}
