#pragma once
#pragma once

// SphereScene.h
//
// A minimal ray-traced scene:
//   - One sphere with a known Disney BRDF material (ground truth)
//   - A solid-colour background acting as the environment light
//
// Usage:
//   1. SphereScene scene;                     // default GT params + geometry
//   2. scene.collectSamples(N_SAMPLES);       // hemisphere sample at the hit point
//   3. Pass scene.samples to the optimizer
//
// Why this is cleaner than the Gaussian splat approach:
//   - Normal is exact (analytic sphere)
//   - Lo is directly ray-traced, no SH bandlimiting
//   - Single point, trivially fast to iterate

#include <vector>
#include <cmath>
#include <iostream>
#include <glm/glm.hpp>

#include "Math.h"
#include "Sampling.h"
#include "BRDFSample.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------------------------------------------------------
// Ground-truth Disney BRDF params
// Chosen for good recoverability:
//   - Strong chromatic baseColor so colour gradient is clear
//   - metallic=0 so diffuse and specular are independent signals
//   - roughness=0.5 so specular lobe is wide enough to sample well
//   - specular=0.5 so both lobes contribute meaningfully
// -------------------------------------------------------------------------
struct GTMaterial {
    glm::vec3 baseColor = glm::vec3(0.8f, 0.3f, 0.1f);
    float     metallic = 0.0f;
    float     roughness = 0.5f;
    float     specular = 0.5f;

    void print() const {
        std::cout << "GT material:\n"
            << "  baseColor = [" << baseColor.r << ", "
            << baseColor.g << ", "
            << baseColor.b << "]\n"
            << "  metallic  = " << metallic << "\n"
            << "  roughness = " << roughness << "\n"
            << "  specular  = " << specular << "\n";
    }
};


// -------------------------------------------------------------------------
// Minimal Disney BRDF evaluation (float, no autodiff)
// Used to compute Lo at the hit point with GT params.
// -------------------------------------------------------------------------
namespace DisneyFloat {

    inline float schlick(float u) {
        float m = glm::clamp(1.f - u, 0.f, 1.f);
        return m * m * m * m * m;
    }

    inline float GTR2(float NdotH, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float t = 1.f + (a2 - 1.f) * NdotH * NdotH;
        return a2 / (float(M_PI) * t * t);
    }

    inline float smithG_GGX(float NdotV, float roughness) {
        float a = roughness * roughness;
        float a2 = a * a;
        float b = NdotV * NdotV;
        return 1.f / (NdotV + std::sqrt(a2 + b - a2 * b));
    }

    inline glm::vec3 evaluate(
        const GTMaterial& mat,
        const glm::vec3& V,   // omega_o: hit ? camera
        const glm::vec3& L,   // omega_i: hit ? light sample
        const glm::vec3& N)
    {
        float ndl = glm::clamp(glm::dot(N, L), 0.f, 1.f);
        float ndv = glm::clamp(glm::dot(N, V), 1e-4f, 1.f);
        if (ndl <= 0.f) return glm::vec3(0.f);

        glm::vec3 H = glm::normalize(L + V);
        float ndh = glm::clamp(glm::dot(N, H), 0.f, 1.f);
        float ldh = glm::clamp(glm::dot(L, H), 0.f, 1.f);

        // Diffuse
        float fd90 = 0.5f + 2.f * ldh * ldh * mat.roughness;
        float FL = schlick(ndl);
        float FV = schlick(ndv);
        float Fd = (1.f + (fd90 - 1.f) * FL) * (1.f + (fd90 - 1.f) * FV);
        glm::vec3 diff = mat.baseColor * (1.f / float(M_PI)) * (1.f - mat.metallic) * Fd;

        // Specular
        glm::vec3 Cspec = (1.f - mat.metallic) * mat.specular * 0.08f
            + mat.metallic * mat.baseColor;
        float D = GTR2(ndh, mat.roughness);
        float FH = schlick(ldh);
        glm::vec3 F = Cspec + (1.f - Cspec) * FH;
        float G = smithG_GGX(ndl, mat.roughness) * smithG_GGX(ndv, mat.roughness);

        return (1.f - mat.metallic) * diff + G * F * D;
    }

} // namespace DisneyFloat


// -------------------------------------------------------------------------
// SphereScene
// -------------------------------------------------------------------------
struct SphereScene {

    // Geometry
    glm::vec3 sphereCenter = glm::vec3(0.f, 0.f, 0.f);
    float     sphereRadius = 2.f;

    // Background: solid colour acting as environment light
    // Bright enough to carry signal; white so all BRDF channels are exercised
    glm::vec3 backgroundColor = glm::vec3(2.f, 2.f, 2.f);

    // Ground-truth material
    GTMaterial material;

    // Camera ray
    // Hits the sphere at a point where N is offset from the camera direction
    // so the half-vector is non-trivial (good for specular recovery).
    glm::vec3 rayOrigin = glm::vec3(0.f, 0.f, -5.f);
    glm::vec3 rayDir = glm::normalize(glm::vec3(-0.3f, 0.3f, 1.f));

    // Derived at construction
    glm::vec3 hitPoint;   // point on sphere surface
    glm::vec3 hitNormal;  // exact analytic normal
    glm::vec3 omega_o;    // view direction (hit ? camera), normalised

    // Collected samples
    std::vector<BRDFSample> samples;

    // -----------------------------------------------------------------------
    // Constructor: find the hit point and compute geometry
    // -----------------------------------------------------------------------
    SphereScene() { computeHitPoint(); }

    void computeHitPoint() {
        // Analytic ray-sphere intersection
        glm::vec3 oc = rayOrigin - sphereCenter;
        float a = glm::dot(rayDir, rayDir);
        float b = 2.f * glm::dot(oc, rayDir);
        float c = glm::dot(oc, oc) - sphereRadius * sphereRadius;
        float disc = b * b - 4.f * a * c;

        if (disc < 0.f) {
            std::cerr << "ERROR: camera ray misses sphere!\n";
            return;
        }

        float t = (-b - std::sqrt(disc)) / (2.f * a);
        hitPoint = rayOrigin + t * rayDir;
        hitNormal = glm::normalize(hitPoint - sphereCenter);
        omega_o = glm::normalize(rayOrigin - hitPoint);  // hit ? camera

        std::cout << "Hit point  : [" << hitPoint.x << ", " << hitPoint.y << ", " << hitPoint.z << "]\n";
        std::cout << "Hit normal : [" << hitNormal.x << ", " << hitNormal.y << ", " << hitNormal.z << "]\n";
        std::cout << "omega_o    : [" << omega_o.x << ", " << omega_o.y << ", " << omega_o.z << "]\n";
    }

    // -----------------------------------------------------------------------
    // traceSecondary
    // Returns the colour seen along direction wi from hitPoint.
    // If it hits the sphere again ? black (self-occlusion, no inter-reflection).
    // Otherwise ? backgroundColor.
    // -----------------------------------------------------------------------
    glm::vec3 traceSecondary(const glm::vec3& wi) const {
        // Ray origin offset along wi to avoid self-intersection
        glm::vec3 orig = hitPoint + hitNormal * 1e-4f;

        glm::vec3 oc = orig - sphereCenter;
        float a = glm::dot(wi, wi);
        float b = 2.f * glm::dot(oc, wi);
        float c = glm::dot(oc, oc) - sphereRadius * sphereRadius;
        float disc = b * b - 4.f * a * c;

        if (disc >= 0.f) {
            float t = (-b - std::sqrt(disc)) / (2.f * a);
            if (t > 1e-4f) return glm::vec3(0.f);  // hits sphere: self-occlusion
        }

        return backgroundColor;  // misses sphere: environment
    }

    // -----------------------------------------------------------------------
    // collectSamples
    //
    // Shoots nSamples cosine-weighted hemisphere rays from hitPoint.
    // For each:
    //   L_i = traceSecondary(wi)          (background colour or 0)
    //   L_o = rendering equation with GT material and ALL samples
    //         (computed analytically after collection so it's exact)
    //
    // Lo is computed once as a high-sample MC estimate, then assigned
    // to every sample as a shared target. This avoids the SH bandlimiting
    // problem entirely: Lo is the direct rendering equation result.
    // -----------------------------------------------------------------------
    void collectSamples(int nSamples = 2048, unsigned seed = 42) {
        samples.clear();
        samples.reserve(nSamples);

        MTRandom rng(seed);

        // Build tangent frame around hitNormal
        Vec3 N_vec = fromGLM(hitNormal);
        Frame frame;
        frame.fromVector(N_vec);

        // Collect hemisphere samples
        for (int i = 0; i < nSamples; ++i) {
            Vec3 localDir = SamplingDistributions::cosineSampleHemisphere(
                rng.next(), rng.next());
            Vec3      wi_vec = frame.toWorld(localDir);
            glm::vec3 wi = wi_vec.ToGlm();

            float ndl = glm::dot(hitNormal, wi);
            if (ndl <= 0.f) continue;

            glm::vec3 Li = traceSecondary(wi);

            BRDFSample s;
            s.splatIndex = 0;
            s.omega_i = wi;
            s.omega_o = omega_o;
            s.normal = hitNormal;
            s.L_i = Li;
            s.L_o = glm::vec3(0.f);  // filled in below
            s.cosTheta = ndl;
            s.weight = 1.f;
            s.shColour = glm::vec3(0.f);
            samples.push_back(s);
        }

        // ---------------------------------------------------------------
        // Compute Lo analytically from the GT material + collected L_i.
        // This is the MC estimate of the rendering equation:
        //   Lo(wo) = (pi / N) * sum[ BRDF(wi, wo, N) * Li ]
        // NdotL cancels with cosine-sampling PDF.
        // Using ALL samples gives a low-variance estimate to use as target.
        // ---------------------------------------------------------------
        glm::vec3 Lo = computeLoGT();

        // Assign the same Lo to every sample — it's the shared target
        for (auto& s : samples)
            s.L_o = Lo;

        std::cout << "Collected " << samples.size() << " samples\n";
        std::cout << "Lo (GT, MC estimate) = ["
            << Lo.r << ", " << Lo.g << ", " << Lo.b << "]\n";
        material.print();
    }

    // -----------------------------------------------------------------------
    // computeLoGT
    // High-sample MC estimate of Lo using the GT material.
    // -----------------------------------------------------------------------
    glm::vec3 computeLoGT() const {
        glm::vec3 sum(0.f);
        int valid = 0;
        for (const auto& s : samples) {
            float ndl = glm::dot(s.normal, s.omega_i);
            if (ndl <= 0.f) continue;
            glm::vec3 brdf = DisneyFloat::evaluate(material, s.omega_o, s.omega_i, s.normal);
            // NdotL cancels with cosine PDF
            sum += brdf * s.L_i;
            ++valid;
        }
        if (valid == 0) return glm::vec3(0.f);
        return (float(M_PI) / float(valid)) * sum;
    }

    // -----------------------------------------------------------------------
    // Print a diagnostic comparing pred(GT) vs Lo
    // If these match, the optimizer WILL converge to GT params.
    // -----------------------------------------------------------------------
    void printSanityCheck() const {
        glm::vec3 Lo = samples.empty() ? glm::vec3(0.f) : samples[0].L_o;
        glm::vec3 pred = computeLoGT();
        std::cout << "\n=== Sanity check ===\n";
        std::cout << "pred(GT) = [" << pred.r << ", " << pred.g << ", " << pred.b << "]\n";
        std::cout << "Lo       = [" << Lo.r << ", " << Lo.g << ", " << Lo.b << "]\n";
        glm::vec3 err = glm::abs(pred - Lo);
        std::cout << "abs err  = [" << err.r << ", " << err.g << ", " << err.b << "]\n";
        std::cout << "(Should be near zero -- Lo was computed from the same samples)\n\n";
    }
};