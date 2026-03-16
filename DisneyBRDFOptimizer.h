#pragma once

// Disney Principled BRDF Optimizer
// Optimizes Disney BRDF parameters from BRDF samples using gradient descent
// 
// Disney BRDF Parameters (11 total):
// - baseColor (vec3): Surface color / albedo
// - metallic (float): 0 = dielectric, 1 = metallic
// - subsurface (float): Subsurface scattering amount
// - specular (float): Specular reflection strength (0-1, default 0.5)
// - roughness (float): Surface roughness (0 = smooth, 1 = rough)
// - specularTint (float): Tint specular toward baseColor
// - anisotropic (float): Anisotropic reflection (0-1)
// - sheen (float): Sheen amount for fabric-like appearance
// - sheenTint (float): Tint sheen toward baseColor
// - clearcoat (float): Clearcoat layer amount
// - clearcoatGloss (float): Clearcoat glossiness

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

// Disney BRDF parameters structure
struct DisneyBRDFParams {
    glm::vec3 baseColor = glm::vec3(0.5f);
    float metallic = 0.0f;
    float subsurface = 0.0f;
    float specular = 0.5f;
    float roughness = 0.5f;
    float specularTint = 0.0f;
    float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.5f;
    float clearcoat = 0.0f;
    float clearcoatGloss = 1.0f;

    // Clamp all parameters to valid ranges
    void clamp() {
        baseColor = glm::clamp(baseColor, glm::vec3(0.0f), glm::vec3(1.0f));
        metallic = glm::clamp(metallic, 0.0f, 1.0f);
        subsurface = glm::clamp(subsurface, 0.0f, 1.0f);
        specular = glm::clamp(specular, 0.0f, 1.0f);
        roughness = glm::clamp(roughness, 0.0f, 1.0f);
        specularTint = glm::clamp(specularTint, 0.0f, 1.0f);
        anisotropic = glm::clamp(anisotropic, 0.0f, 1.0f);
        sheen = glm::clamp(sheen, 0.0f, 1.0f);
        sheenTint = glm::clamp(sheenTint, 0.0f, 1.0f);
        clearcoat = glm::clamp(clearcoat, 0.0f, 1.0f);
        clearcoatGloss = glm::clamp(clearcoatGloss, 0.0f, 1.0f);
    }
};

// Helper functions for Disney BRDF evaluation
namespace DisneyBRDF {

    // Schlick Fresnel approximation
    inline float schlickFresnel(float u) {
        float m = glm::clamp(1.0f - u, 0.0f, 1.0f);
        float m2 = m * m;
        return m2 * m2 * m; // m^5
    }

    // GTR1 (for clearcoat)
    inline float GTR1(float NdotH, float a) {
        if (a >= 1.0f) return 1.0f / M_PI;
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
        return (a2 - 1.0f) / (M_PI * std::log(a2) * t);
    }

    // GTR2 (GGX/Trowbridge-Reitz)
    inline float GTR2(float NdotH, float a) {
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
        return a2 / (M_PI * t * t);
    }

    // GTR2 Anisotropic
    inline float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay) {
        float denom = HdotX * HdotX / (ax * ax) + HdotY * HdotY / (ay * ay) + NdotH * NdotH;
        return 1.0f / (M_PI * ax * ay * denom * denom);
    }

    // Smith GGX shadowing function
    inline float smithG_GGX(float NdotV, float alphaG) {
        float a = alphaG * alphaG;
        float b = NdotV * NdotV;
        return 1.0f / (NdotV + std::sqrt(a + b - a * b));
    }

    // Disney diffuse (Burley)
    inline float disney_diffuse(float NdotL, float NdotV, float LdotH, float roughness) {
        float fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
        float FL = schlickFresnel(NdotL);
        float FV = schlickFresnel(NdotV);
        return (1.0f + (fd90 - 1.0f) * FL) * (1.0f + (fd90 - 1.0f) * FV);
    }

    // Evaluate Disney BRDF
    glm::vec3 evaluate(
        const DisneyBRDFParams& params,
        const glm::vec3& V,  // View direction (toward camera)
        const glm::vec3& L,  // Light direction (toward light)
        const glm::vec3& N,  // Surface normal
        const glm::vec3& X,  // Tangent
        const glm::vec3& Y   // Bitangent
    ) {
        float NdotL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
        float NdotV = glm::clamp(glm::dot(N, V), 0.0f, 1.0f);
        
        if (NdotL <= 0.0f || NdotV <= 0.0f) return glm::vec3(0.0f);

        glm::vec3 H = glm::normalize(L + V);
        float NdotH = glm::clamp(glm::dot(N, H), 0.0f, 1.0f);
        float LdotH = glm::clamp(glm::dot(L, H), 0.0f, 1.0f);

        // Luminance of base color
        float Cdlum = 0.3f * params.baseColor.r + 0.6f * params.baseColor.g + 0.1f * params.baseColor.b;

        // Normalize lum. to isolate hue+sat
        glm::vec3 Ctint = Cdlum > 0.0f ? params.baseColor / Cdlum : glm::vec3(1.0f);
        
        // Specular color
        glm::vec3 Cspec0 = glm::mix(
            params.specular * 0.08f * glm::mix(glm::vec3(1.0f), Ctint, params.specularTint),
            params.baseColor,
            params.metallic
        );

        // Sheen color
        glm::vec3 Csheen = glm::mix(glm::vec3(1.0f), Ctint, params.sheenTint);

        // --- Diffuse ---
        float Fd = disney_diffuse(NdotL, NdotV, LdotH, params.roughness);
        
        // Subsurface approximation (Hanrahan-Krueger)
        float Fss90 = LdotH * LdotH * params.roughness;
        float Fss = glm::mix(1.0f, Fss90, schlickFresnel(NdotL)) * 
                    glm::mix(1.0f, Fss90, schlickFresnel(NdotV));
        float ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - 0.5f) + 0.5f);

        // Diffuse component
        glm::vec3 diffuse = (1.0f / M_PI) * glm::mix(Fd, ss, params.subsurface) * params.baseColor;

        // --- Specular ---
        float aspect = std::sqrt(1.0f - params.anisotropic * 0.9f);
        float ax = std::max(0.001f, params.roughness * params.roughness / aspect);
        float ay = std::max(0.001f, params.roughness * params.roughness * aspect);
        
        float Ds;
        if (params.anisotropic > 0.0f) {
            float HdotX = glm::dot(H, X);
            float HdotY = glm::dot(H, Y);
            Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay);
        } else {
            Ds = GTR2(NdotH, ax);
        }

        float FH = schlickFresnel(LdotH);
        glm::vec3 Fs = glm::mix(Cspec0, glm::vec3(1.0f), FH);
        
        float Gs = smithG_GGX(NdotL, params.roughness) * smithG_GGX(NdotV, params.roughness);

        // Sheen
        glm::vec3 Fsheen = FH * params.sheen * Csheen;

        // Specular component
        glm::vec3 specular = Gs * Fs * Ds;

        // --- Clearcoat ---
        float Dr = GTR1(NdotH, glm::mix(0.1f, 0.001f, params.clearcoatGloss));
        float Fr = glm::mix(0.04f, 1.0f, FH);
        float Gr = smithG_GGX(NdotL, 0.25f) * smithG_GGX(NdotV, 0.25f);
        glm::vec3 clearcoat = glm::vec3(0.25f * params.clearcoat * Gr * Fr * Dr);

        // Combine all components
        return ((1.0f - params.metallic) * (diffuse + Fsheen) + specular + clearcoat);
    }
}

// Gradient structure for all Disney BRDF parameters
struct DisneyBRDFGradient {
    glm::vec3 baseColor = glm::vec3(0.0f);
    float metallic = 0.0f;
    float subsurface = 0.0f;
    float specular = 0.0f;
    float roughness = 0.0f;
    float specularTint = 0.0f;
    float anisotropic = 0.0f;
    float sheen = 0.0f;
    float sheenTint = 0.0f;
    float clearcoat = 0.0f;
    float clearcoatGloss = 0.0f;

    void reset() {
        baseColor = glm::vec3(0.0f);
        metallic = subsurface = specular = roughness = 0.0f;
        specularTint = anisotropic = sheen = sheenTint = 0.0f;
        clearcoat = clearcoatGloss = 0.0f;
    }

    DisneyBRDFGradient& operator+=(const DisneyBRDFGradient& other) {
        baseColor += other.baseColor;
        metallic += other.metallic;
        subsurface += other.subsurface;
        specular += other.specular;
        roughness += other.roughness;
        specularTint += other.specularTint;
        anisotropic += other.anisotropic;
        sheen += other.sheen;
        sheenTint += other.sheenTint;
        clearcoat += other.clearcoat;
        clearcoatGloss += other.clearcoatGloss;
        return *this;
    }

    DisneyBRDFGradient operator*(float s) const {
        DisneyBRDFGradient result = *this;
        result.baseColor *= s;
        result.metallic *= s;
        result.subsurface *= s;
        result.specular *= s;
        result.roughness *= s;
        result.specularTint *= s;
        result.anisotropic *= s;
        result.sheen *= s;
        result.sheenTint *= s;
        result.clearcoat *= s;
        result.clearcoatGloss *= s;
        return result;
    }
};

// Compute numerical gradient using finite differences
DisneyBRDFGradient computeNumericalGradient(
    const DisneyBRDFParams& params,
    const BRDFSample& sample,
    const glm::vec3& tangent,
    const glm::vec3& bitangent,
    float epsilon = 1e-4f
) {
    DisneyBRDFGradient grad;
    
    // Evaluate current loss
    glm::vec3 f0 = DisneyBRDF::evaluate(params, sample.omega_o, sample.omega_i, 
                                        sample.normal, tangent, bitangent);
    glm::vec3 residual0 = (f0 * sample.L_i * sample.cosTheta) - sample.L_o;
    float loss0 = glm::dot(residual0, residual0);

    // Helper lambda for finite difference
    auto finiteDiff = [&](float& param, float& gradOut) {
        float original = param;
        param = original + epsilon;
        glm::vec3 f1 = DisneyBRDF::evaluate(params, sample.omega_o, sample.omega_i,
                                            sample.normal, tangent, bitangent);
        glm::vec3 residual1 = (f1 * sample.L_i * sample.cosTheta) - sample.L_o;
        float loss1 = glm::dot(residual1, residual1);
        gradOut = (loss1 - loss0) / epsilon;
        param = original;
    };

    // Compute gradients for each parameter
    DisneyBRDFParams temp = params;
    
    // BaseColor (per channel)
    for (int i = 0; i < 3; ++i) {
        float original = temp.baseColor[i];
        temp.baseColor[i] = original + epsilon;
        glm::vec3 f1 = DisneyBRDF::evaluate(temp, sample.omega_o, sample.omega_i,
                                            sample.normal, tangent, bitangent);
        glm::vec3 residual1 = (f1 * sample.L_i * sample.cosTheta) - sample.L_o;
        float loss1 = glm::dot(residual1, residual1);
        grad.baseColor[i] = (loss1 - loss0) / epsilon;
        temp.baseColor[i] = original;
    }

    finiteDiff(temp.metallic, grad.metallic);
    finiteDiff(temp.subsurface, grad.subsurface);
    finiteDiff(temp.specular, grad.specular);
    finiteDiff(temp.roughness, grad.roughness);
    finiteDiff(temp.specularTint, grad.specularTint);
    finiteDiff(temp.anisotropic, grad.anisotropic);
    finiteDiff(temp.sheen, grad.sheen);
    finiteDiff(temp.sheenTint, grad.sheenTint);
    finiteDiff(temp.clearcoat, grad.clearcoat);
    finiteDiff(temp.clearcoatGloss, grad.clearcoatGloss);

    return grad;
}

// Main Disney BRDF optimization function
void optimizeDisneyBRDF(
    const std::vector<BRDFSample>& samples,
    std::vector<Gaussian>& gaussians,
    int maxIterations = 1000,
    float learningRate = 0.001f,
    bool optimizeMetallic = true,
    bool optimizeRoughness = true,
    bool optimizeSpecular = true,
    bool optimizeAdvanced = false  // Subsurface, anisotropic, sheen, clearcoat
) {
    std::cout << "Starting Disney BRDF optimization...\n";
    std::cout << "Max iterations: " << maxIterations << ", Learning rate: " << learningRate << "\n";

    // Group samples by splat index
    std::unordered_map<int, std::vector<const BRDFSample*>> groups;
    for (const auto& s : samples) {
        groups[s.splatIndex].push_back(&s);
    }

    size_t maxIndex = 0;
    for (const auto& kv : groups) {
        maxIndex = std::max<size_t>(maxIndex, (size_t)kv.first);
    }

    // Initialize parameters for each splat
    std::vector<DisneyBRDFParams> params(maxIndex + 1);
    
    // Initialize with reasonable defaults
    for (auto& p : params) {
        p.baseColor = glm::vec3(0.5f);
        p.metallic = 0.0f;
        p.roughness = 0.5f;
        p.specular = 0.5f;
    }

    std::cout << "Optimizing " << groups.size() << " splats...\n";

    int progressInterval = std::max(1, (int)groups.size() / 10);
    int processed = 0;

    // Optimize each splat independently
    for (const auto& [splatIdx, sampleList] : groups) {
        if (sampleList.empty()) continue;

        DisneyBRDFParams& p = params[splatIdx];
        
        // Compute tangent frame (simplified - assumes normal is up-ish)
        glm::vec3 normal = sampleList[0]->normal;
        glm::vec3 tangent, bitangent;
        if (std::abs(normal.z) > 0.9f) {
            tangent = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            tangent = glm::normalize(glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), normal));
        }
        bitangent = glm::cross(normal, tangent);

        // Optimization loop
        for (int iter = 0; iter < maxIterations; ++iter) {
            DisneyBRDFGradient totalGrad;
            totalGrad.reset();
            float totalLoss = 0.0f;

            // Accumulate gradients from all samples
            for (const BRDFSample* sample : sampleList) {
                DisneyBRDFGradient grad = computeNumericalGradient(p, *sample, tangent, bitangent);
                totalGrad += grad;

                // Compute loss for monitoring
                glm::vec3 f = DisneyBRDF::evaluate(p, sample->omega_o, sample->omega_i,
                                                   sample->normal, tangent, bitangent);
                glm::vec3 residual = (f * sample->L_i * sample->cosTheta) - sample->L_o;
                totalLoss += glm::dot(residual, residual);
            }

            // Average gradients
            float n = static_cast<float>(sampleList.size());
            totalGrad = totalGrad * (1.0f / n);
            totalLoss /= n;

            // Gradient descent update
            p.baseColor -= learningRate * totalGrad.baseColor;
            if (optimizeMetallic) p.metallic -= learningRate * totalGrad.metallic;
            if (optimizeRoughness) p.roughness -= learningRate * totalGrad.roughness;
            if (optimizeSpecular) p.specular -= learningRate * totalGrad.specular;
            
            if (optimizeAdvanced) {
                p.subsurface -= learningRate * totalGrad.subsurface;
                p.specularTint -= learningRate * totalGrad.specularTint;
                p.anisotropic -= learningRate * totalGrad.anisotropic;
                p.sheen -= learningRate * totalGrad.sheen;
                p.sheenTint -= learningRate * totalGrad.sheenTint;
                p.clearcoat -= learningRate * totalGrad.clearcoat;
                p.clearcoatGloss -= learningRate * totalGrad.clearcoatGloss;
            }

            // Clamp parameters
            p.clamp();

            // Early stopping if converged
            if (iter > 10 && totalLoss < 1e-6f) break;
        }

        processed++;
        if (processed % progressInterval == 0) {
            std::cout << "Progress: " << (processed * 100 / groups.size()) << "%\r" << std::flush;
        }
    }

    std::cout << "\nOptimization complete!\n";

    // Write results to CSV
    std::ofstream out("disney_brdf_params.csv");
    out << "splatIndex,baseColor.r,baseColor.g,baseColor.b,metallic,subsurface,specular,roughness,";
    out << "specularTint,anisotropic,sheen,sheenTint,clearcoat,clearcoatGloss,sampleCount\n";
    
    for (size_t i = 0; i < params.size(); ++i) {
        auto it = groups.find((int)i);
        if (it == groups.end()) continue;
        
        const DisneyBRDFParams& p = params[i];
        out << i << ","
            << p.baseColor.r << "," << p.baseColor.g << "," << p.baseColor.b << ","
            << p.metallic << "," << p.subsurface << "," << p.specular << "," << p.roughness << ","
            << p.specularTint << "," << p.anisotropic << "," << p.sheen << "," << p.sheenTint << ","
            << p.clearcoat << "," << p.clearcoatGloss << "," << it->second.size() << "\n";
    }
    out.close();

    // Update gaussians with optimized base color (as albedo)
    for (size_t i = 0; i < params.size(); ++i) {
        if (i < gaussians.size()) {
            gaussians[i].testAlbedo = params[i].baseColor;
        }
    }

    std::cout << "Results saved to disney_brdf_params.csv\n";
}
