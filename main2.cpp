//#include <iostream>
//#include <fstream>
//#include <set>
//#include <thread>
//
//#include "happly.h"
//#include "GamesEngineeringBase.h"
//
//#include "Math.h"
//#include "Sampling.h"
//#include "Imaging.h"
//
//#include "BRDFOptimizerSamples.h"
//#include "SingleOptim.h"
//#include "DisneyBRDFOptimizerSimple.h"
//#include "CosBRDF_Disney.h"
//
//#include "BRDF_Optim_AutoDiff.h"
//#include "SpherePointOptim.h"
//
//// ==========================================================================
////  Spherical Gaussian (SG) support
////
////  Each SG lobe encodes one component of Lo(wo):
////
////    G(wo; dir, lambda, amp) = amp * exp( lambda * (dot(wo, dir) - 1) )
////
////  The full outgoing radiance is the sum of K_SG lobes:
////
////    Lo(wo) = sum_{k=0}^{K-1}  amp_k * exp( lambda_k * (dot(wo, dir_k) - 1) )
////
////  Properties stored in the PLY  (per vertex, K_SG lobes):
////    sg_{k}_dir_x, sg_{k}_dir_y, sg_{k}_dir_z
////    sg_{k}_lambda
////    sg_{k}_amp_r,  sg_{k}_amp_g,  sg_{k}_amp_b
////
////  Key advantage over SH L3:
////    - SH L3 (16 bands) minimum angular feature: ~36 deg
////    - A single SG with lambda=50 covers a ~20-deg lobe (roughness=0.3 specular)
////    - 5 SGs can represent diffuse + specular with far less approximation error
//// ==========================================================================
//
//struct SphGaussian {
//    glm::vec3 dir;     // unit direction (lobe centre)
//    float     lambda;  // sharpness (larger = narrower lobe)
//    glm::vec3 amp;     // RGB amplitude
//};
//
//// Per-splat SG lobes.  allSGs[i] = SG lobes for Gaussian splat i.
//// Populated by parsePLYWithSG and read by collectSamplesForView_SG.
//std::vector<std::vector<SphGaussian>> allSGs;
//
//constexpr int K_SG_DEFAULT = 5;
//
//// ==========================================================================
////  Standard SH constants and helpers (unchanged from main.cpp)
//// ==========================================================================
//#define SH_C0 0.28209479177387814f
//#define SH_C1 0.4886025119029199f
//
//#define SH_C2_0  1.0925484305920792f
//#define SH_C2_1 -1.0925484305920792f
//#define SH_C2_2  0.31539156525252005f
//#define SH_C2_3 -1.0925484305920792f
//#define SH_C2_4  0.5462742152960396f
//
//#define SH_C3_0 -0.5900435899266435f
//#define SH_C3_1  2.890611442640554f
//#define SH_C3_2 -0.4570457994644658f
//#define SH_C3_3  0.3731763325901154f
//#define SH_C3_4 -0.4570457994644658f
//#define SH_C3_5  1.445305721320277f
//#define SH_C3_6 -0.5900435899266435f
//
//#define tileSize 16
//
//
//std::vector<BRDFSample> BRDFSampleList;
//std::vector<std::vector<BRDFSample>> BRDFSampleList_vec;
//
//
//// --------------------------------------------------------------------------
//// evaluateSGLo
////
//// Evaluates Lo(wo) as a sum of Spherical Gaussian lobes:
////   Lo(wo) = clamp( sum_k amp_k * exp(lambda_k * (dot(wo, dir_k) - 1)), 0, inf )
////
//// All amplitudes stored in the PLY are non-negative (NNLS fitting guarantees
//// this), so the sum is always >= 0 — clamping is just a safety guard.
//// --------------------------------------------------------------------------
//glm::vec3 evaluateSGLo(const glm::vec3& wo, const std::vector<SphGaussian>& sgs)
//{
//    glm::vec3 Lo(0.f);
//    for (const SphGaussian& sg : sgs) {
//        float dot   = glm::dot(wo, sg.dir);
//        float basis = std::exp(sg.lambda * (dot - 1.f));
//        Lo += sg.amp * basis;
//    }
//    return glm::max(Lo, glm::vec3(0.f));
//}
//
//
//// --------------------------------------------------------------------------
//// parsePLYWithSG
////
//// Extends parsePLY by additionally reading K_sg SG lobes per vertex from
//// the extended PLY written by SphereSGGen.py.
////
//// Falls back gracefully when SG properties are absent (e.g., old PLY files):
//// allSGs is left empty / all-zero in that case, and evaluateSGLo returns 0.
//// --------------------------------------------------------------------------
//void parsePLYWithSG(const std::string& filename,
//                    std::vector<Gaussian>& gaussians,
//                    int K_sg = K_SG_DEFAULT)
//{
//    // ------------------------------------------------------------------
//    // Step 1: standard Gaussian parse (positions, normals, SH, scale, rot)
//    // ------------------------------------------------------------------
//    happly::PLYData plyIn(filename.c_str());
//
//    std::vector<float> prop1 = plyIn.getElement("vertex").getProperty<float>("x");
//    std::vector<float> prop2 = plyIn.getElement("vertex").getProperty<float>("y");
//    std::vector<float> prop3 = plyIn.getElement("vertex").getProperty<float>("z");
//
//    int size = (int)prop1.size();
//    gaussians = std::vector<Gaussian>(size);
//
//    for (int i = 0; i < size; i++) {
//        gaussians[i].pos   = Vec3(prop1[i], prop2[i], prop3[i]);
//        gaussians[i].index = i;
//    }
//
//    // Normals (stored in same file)
//    {
//        std::vector<double> nx = plyIn.getElement("vertex").getProperty<double>("nx");
//        std::vector<double> ny = plyIn.getElement("vertex").getProperty<double>("ny");
//        std::vector<double> nz = plyIn.getElement("vertex").getProperty<double>("nz");
//        for (int i = 0; i < size; i++)
//            gaussians[i].GaussNormal = glm::vec3(nx[i], ny[i], nz[i]);
//    }
//
//    // DC SH -> base_color for warm-start
//    prop1 = plyIn.getElement("vertex").getProperty<float>("f_dc_0");
//    prop2 = plyIn.getElement("vertex").getProperty<float>("f_dc_1");
//    prop3 = plyIn.getElement("vertex").getProperty<float>("f_dc_2");
//
//    for (int i = 0; i < size; i++) {
//        gaussians[i].ZeroSH = Vec3(prop1[i], prop2[i], prop3[i]);
//        // color = DC * SH_C0 + 0.5 = base_color (SphereSGGen pins DC to base_color)
//        Vec3 c = gaussians[i].ZeroSH * SH_C0;
//        Colour col;
//        col += c;
//        col += Vec3(0.5f);
//        gaussians[i].color     = col;
//        gaussians[i].testColor = col.ToGlm();
//    }
//
//    // Scale
//    prop1 = plyIn.getElement("vertex").getProperty<float>("scale_0");
//    prop2 = plyIn.getElement("vertex").getProperty<float>("scale_1");
//    prop3 = plyIn.getElement("vertex").getProperty<float>("scale_2");
//    for (int i = 0; i < size; i++) {
//        gaussians[i].scale = Vec3(prop1[i], prop2[i], prop3[i]);
//        gaussians[i].compute_gaussian_aabb();
//    }
//
//    // Opacity
//    std::vector<float> prop4 = plyIn.getElement("vertex").getProperty<float>("opacity");
//    for (int i = 0; i < size; i++)
//        gaussians[i].opacity = sigmoid(prop4[i]);
//
//    // Rotation
//    prop1 = plyIn.getElement("vertex").getProperty<float>("rot_0");
//    prop2 = plyIn.getElement("vertex").getProperty<float>("rot_1");
//    prop3 = plyIn.getElement("vertex").getProperty<float>("rot_2");
//    prop4 = plyIn.getElement("vertex").getProperty<float>("rot_3");
//    for (int i = 0; i < size; i++) {
//        gaussians[i].rotation = Vec3(prop1[i], prop2[i], prop3[i], prop4[i]);
//        gaussians[i].compute_gaussian_covariance();
//    }
//
//    // Higher SH (f_rest) – still present in the file for compatibility,
//    // but in SG mode they are all-zero and NOT used for Lo evaluation.
//    for (int b = 0; b < 45; b++) {
//        std::string name = "f_rest_" + std::to_string(b);
//        prop1 = plyIn.getElement("vertex").getProperty<float>(name);
//        for (int j = 0; j < size; j++)
//            gaussians[j].higherSH.push_back(prop1[j]);
//    }
//
//    // ------------------------------------------------------------------
//    // Step 2: read SG lobes
//    // ------------------------------------------------------------------
//    allSGs.assign(size, {});
//
//    bool any_sg_found = false;
//    for (int k = 0; k < K_sg; k++) {
//        std::string pfx = "sg_" + std::to_string(k) + "_";
//
//        // Check that all seven properties exist before reading
//        bool have_all =
//            plyIn.getElement("vertex").hasProperty(pfx + "dir_x") &&
//            plyIn.getElement("vertex").hasProperty(pfx + "dir_y") &&
//            plyIn.getElement("vertex").hasProperty(pfx + "dir_z") &&
//            plyIn.getElement("vertex").hasProperty(pfx + "lambda") &&
//            plyIn.getElement("vertex").hasProperty(pfx + "amp_r") &&
//            plyIn.getElement("vertex").hasProperty(pfx + "amp_g") &&
//            plyIn.getElement("vertex").hasProperty(pfx + "amp_b");
//
//        if (!have_all) {
//            std::cerr << "  WARNING: SG lobe " << k
//                      << " properties missing in PLY. "
//                      << "Run SphereSGGen.py first.\n";
//            continue;
//        }
//        any_sg_found = true;
//
//        auto dir_x = plyIn.getElement("vertex").getProperty<float>(pfx + "dir_x");
//        auto dir_y = plyIn.getElement("vertex").getProperty<float>(pfx + "dir_y");
//        auto dir_z = plyIn.getElement("vertex").getProperty<float>(pfx + "dir_z");
//        auto lam   = plyIn.getElement("vertex").getProperty<float>(pfx + "lambda");
//        auto amp_r = plyIn.getElement("vertex").getProperty<float>(pfx + "amp_r");
//        auto amp_g = plyIn.getElement("vertex").getProperty<float>(pfx + "amp_g");
//        auto amp_b = plyIn.getElement("vertex").getProperty<float>(pfx + "amp_b");
//
//        for (int i = 0; i < size; i++) {
//            allSGs[i].push_back(SphGaussian{
//                glm::normalize(glm::vec3(dir_x[i], dir_y[i], dir_z[i])),
//                lam[i],
//                glm::max(glm::vec3(amp_r[i], amp_g[i], amp_b[i]), glm::vec3(0.f))
//            });
//        }
//    }
//
//    if (any_sg_found)
//        std::cout << "  Loaded " << size << " Gaussians with "
//                  << K_sg << " SG lobes each.\n";
//    else
//        std::cout << "  WARNING: No SG lobes found. Lo will be zero.\n";
//}
//
//
//// ==========================================================================
////  All functions below are identical to main.cpp except where marked [SG]
//// ==========================================================================
//
//void writeBRDFSamples(const std::string& filename, std::vector<Gaussian>& gaussians) {
//    std::ofstream file(filename);
//    if (!file.is_open()) return;
//
//    file << "splatIndex,omega_i.x,omega_i.y,omega_i.z,omega_o.x,omega_o.y,omega_o.z,"
//            "normal.x,normal.y,normal.z,L_i.x,L_i.y,L_i.z,L_o.x,L_o.y,L_o.z,"
//            "cosTheta,weight,SH.r,SH.g,SH.b,albedo.r,albedo.g,albedo.b\n";
//
//    for (const auto& s : BRDFSampleList) {
//        file << s.splatIndex << ","
//            << s.omega_i.x << "," << s.omega_i.y << "," << s.omega_i.z << ","
//            << s.omega_o.x << "," << s.omega_o.y << "," << s.omega_o.z << ","
//            << s.normal.x  << "," << s.normal.y  << "," << s.normal.z  << ","
//            << s.L_i.x     << "," << s.L_i.y     << "," << s.L_i.z     << ","
//            << s.L_o.x     << "," << s.L_o.y     << "," << s.L_o.z     << ","
//            << s.cosTheta  << ","
//            << s.weight    << ","
//            << s.shColour.x << "," << s.shColour.y << "," << s.shColour.z << ","
//            << gaussians[s.splatIndex].color.r << ","
//            << gaussians[s.splatIndex].color.g << ","
//            << gaussians[s.splatIndex].color.b << "\n";
//    }
//    file << "\n\n";
//    file.close();
//    std::cout << "Saved " << BRDFSampleList.size()
//              << " BRDF samples to " << filename << std::endl;
//}
//
//
//Colour evaluateSphericalHarmonics(const Vec3& viewDir, Gaussian& gaussian) {
//    Vec3 dir = viewDir.normalize();
//    float x = dir.x, y = dir.y, z = dir.z;
//    float xx = x*x, yy = y*y, zz = z*z;
//    float xy = x*y, xz = x*z, yz = y*z;
//
//    Vec3 color = gaussian.ZeroSH * SH_C0;
//    color = color
//        - (Vec3(gaussian.higherSH[0],  gaussian.higherSH[1],  gaussian.higherSH[2])  * SH_C1 * y)
//        + (Vec3(gaussian.higherSH[3],  gaussian.higherSH[4],  gaussian.higherSH[5])  * SH_C1 * z)
//        - (Vec3(gaussian.higherSH[6],  gaussian.higherSH[7],  gaussian.higherSH[8])  * SH_C1 * x);
//    color = color
//        + (Vec3(gaussian.higherSH[9],  gaussian.higherSH[10], gaussian.higherSH[11]) * SH_C2_0 * xy)
//        + (Vec3(gaussian.higherSH[12], gaussian.higherSH[13], gaussian.higherSH[14]) * SH_C2_1 * yz)
//        + (Vec3(gaussian.higherSH[15], gaussian.higherSH[16], gaussian.higherSH[17]) * SH_C2_2 * (2.f*zz - xx - yy))
//        + (Vec3(gaussian.higherSH[18], gaussian.higherSH[19], gaussian.higherSH[20]) * SH_C2_3 * xz)
//        + (Vec3(gaussian.higherSH[21], gaussian.higherSH[22], gaussian.higherSH[23]) * SH_C2_4 * (xx - yy));
//    color = color
//        + (Vec3(gaussian.higherSH[24], gaussian.higherSH[25], gaussian.higherSH[26]) * SH_C3_0 * y * (3.f*xx - yy))
//        + (Vec3(gaussian.higherSH[27], gaussian.higherSH[28], gaussian.higherSH[29]) * SH_C3_1 * xy * z)
//        + (Vec3(gaussian.higherSH[30], gaussian.higherSH[31], gaussian.higherSH[32]) * SH_C3_2 * y * (4.f*zz - xx - yy))
//        + (Vec3(gaussian.higherSH[33], gaussian.higherSH[34], gaussian.higherSH[35]) * SH_C3_3 * z * (2.f*zz - 3.f*xx - 3.f*yy))
//        + (Vec3(gaussian.higherSH[36], gaussian.higherSH[37], gaussian.higherSH[38]) * SH_C3_4 * x * (4.f*zz - xx - yy))
//        + (Vec3(gaussian.higherSH[39], gaussian.higherSH[40], gaussian.higherSH[41]) * SH_C3_5 * z * (xx - yy))
//        + (Vec3(gaussian.higherSH[42], gaussian.higherSH[43], gaussian.higherSH[44]) * SH_C3_6 * x * (xx - 3.f*yy));
//
//    Colour c;
//    c += color;
//    c += Vec3(0.5f);
//    return c;
//}
//
//Colour viewIndependent(Gaussian& gaussian) {
//    Vec3 color = gaussian.ZeroSH * SH_C0;
//    Colour c;
//    c += color;
//    c += Vec3(0.5f);
//    return c;
//}
//
//
//void parsePLY(std::string filename, std::vector<Gaussian>& gaussians, std::string Normalfilename) {
//    happly::PLYData plyIn(filename.c_str());
//    std::vector<float> prop1 = plyIn.getElement("vertex").getProperty<float>("x");
//    std::vector<float> prop2 = plyIn.getElement("vertex").getProperty<float>("y");
//    std::vector<float> prop3 = plyIn.getElement("vertex").getProperty<float>("z");
//
//    int size = prop1.size();
//    gaussians = std::vector<Gaussian>(size);
//
//    for (size_t i = 0; i < prop1.size(); i++) {
//        gaussians[i].pos = Vec3(prop1[i], prop2[i], prop3[i]);
//        gaussians[i].index = i;
//    }
//
//    happly::PLYData plyInNorm(Normalfilename.c_str());
//    std::vector<double> nx = plyInNorm.getElement("vertex").getProperty<double>("nx");
//    std::vector<double> ny = plyInNorm.getElement("vertex").getProperty<double>("ny");
//    std::vector<double> nz = plyInNorm.getElement("vertex").getProperty<double>("nz");
//
//    for (size_t i = 0; i < prop1.size(); i++)
//        gaussians[i].GaussNormal = glm::vec3(nx[i], ny[i], nz[i]);
//
//    prop1 = plyIn.getElement("vertex").getProperty<float>("f_dc_0");
//    prop2 = plyIn.getElement("vertex").getProperty<float>("f_dc_1");
//    prop3 = plyIn.getElement("vertex").getProperty<float>("f_dc_2");
//
//    for (size_t i = 0; i < prop1.size(); i++) {
//        gaussians[i].ZeroSH = Vec3(prop1[i], prop2[i], prop3[i]);
//        gaussians[i].color  = viewIndependent(gaussians[i]);
//        gaussians[i].testColor = gaussians[i].color.ToGlm();
//    }
//
//    prop1 = plyIn.getElement("vertex").getProperty<float>("scale_0");
//    prop2 = plyIn.getElement("vertex").getProperty<float>("scale_1");
//    prop3 = plyIn.getElement("vertex").getProperty<float>("scale_2");
//
//    for (size_t i = 0; i < prop1.size(); i++) {
//        gaussians[i].scale = Vec3(prop1[i], prop2[i], prop3[i]);
//        gaussians[i].compute_gaussian_aabb();
//    }
//
//    std::vector<float> prop4 = plyIn.getElement("vertex").getProperty<float>("opacity");
//    for (size_t i = 0; i < prop1.size(); i++)
//        gaussians[i].opacity = sigmoid(prop4[i]);
//
//    prop1 = plyIn.getElement("vertex").getProperty<float>("rot_0");
//    prop2 = plyIn.getElement("vertex").getProperty<float>("rot_1");
//    prop3 = plyIn.getElement("vertex").getProperty<float>("rot_2");
//    prop4 = plyIn.getElement("vertex").getProperty<float>("rot_3");
//
//    for (size_t i = 0; i < prop1.size(); i++) {
//        gaussians[i].rotation = Vec3(prop1[i], prop2[i], prop3[i], prop4[i]);
//        gaussians[i].compute_gaussian_covariance();
//    }
//
//    for (int i = 0; i < 45; i++) {
//        std::string name = "f_rest_" + std::to_string(i);
//        prop1 = plyIn.getElement("vertex").getProperty<float>(name);
//        for (size_t j = 0; j < prop1.size(); j++)
//            gaussians[j].higherSH.push_back(prop1[j]);
//    }
//}
//
//
//Colour GaussianColor(Ray& ray, std::vector<Gaussian>& in, int& contribution_count)
//{
//    struct Hit { float t; Gaussian* g; };
//    std::vector<Hit> hits; hits.reserve(in.size());
//    contribution_count = 0;
//
//    for (auto& g : in) {
//        float t = ray.dir.dot((g.pos - ray.o));
//        if (t <= 0) continue;
//        hits.push_back({ t, &g });
//    }
//    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
//        return a.t < b.t; });
//
//    Colour color(0, 0, 0);
//    float tr = 1.f;
//
//    for (auto& h : hits) {
//        Gaussian& g = *h.g;
//        float alpha = g.computeAlpha(ray);
//        if (tr < 0.001f) break;
//        Vec3   viewDir = (ray.o - g.pos).normalize();
//        Colour SHColor = evaluateSphericalHarmonics(viewDir, g);
//        float contribution = alpha * tr;
//        if (contribution > 0.05f) {
//            color = color + (SHColor * alpha * tr);
//            tr *= (1.0f - alpha);
//            contribution_count++;
//        }
//    }
//    return color;
//}
//
//Colour GaussianAlbedo(Ray& ray, std::vector<Gaussian>& in)
//{
//    struct Hit { float t; Gaussian* g; };
//    std::vector<Hit> hits; hits.reserve(in.size());
//
//    for (auto& g : in) {
//        float t = ray.dir.dot((g.pos - ray.o));
//        if (t <= 0) continue;
//        hits.push_back({ t, &g });
//    }
//    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
//        return a.t < b.t; });
//
//    Colour color(0, 0, 0);
//    float tr = 1.f;
//
//    for (auto& h : hits) {
//        Gaussian& g = *h.g;
//        float alpha = g.computeAlpha(ray);
//        if (tr < 0.001f) break;
//        Colour SHColor = fromGLMC(g.testAlbedo);
//        color = color + (SHColor * alpha * tr);
//        tr *= (1.0f - alpha);
//    }
//    color.correct();
//    return color;
//}
//
//
//void monteCarloSampling(
//    MTRandom& Sampler,
//    Gaussian& g,
//    std::vector<Gaussian>& all,
//    BVHNode* bvh,
//    const glm::vec3& omega_o,
//    std::vector<BRDFSample>& out,
//    float contribution,
//    int threadID = 0)
//{
//    const int N_SAMPLES = 128;
//
//    Vec3 normal = fromGLM(g.GaussNormal);
//    if (normal.dot(fromGLM(omega_o)) < 0.0f)
//        normal = normal * -1.f;
//
//    glm::vec3 normal_glm = normal.ToGlm();
//    Frame frame;
//    frame.fromVector(normal);
//
//    for (int s = 0; s < N_SAMPLES; ++s) {
//        Vec3 localDir = SamplingDistributions::cosineSampleHemisphere(
//            Sampler.next(), Sampler.next());
//        Vec3 omega_i_world = frame.toWorld(localDir);
//
//        float NdotL = normal.dot(omega_i_world);
//        if (NdotL <= 0.0f) continue;
//
//        Ray newRay;
//        newRay.init(g.pos + (omega_i_world * EPSILON), omega_i_world);
//        bvh->traverse(newRay, all, threadID + threadNum * 2);
//
//        int contribution_count = 0;
//        Colour LiColor = GaussianColor(
//            newRay,
//            bvh->getIntersectedGaussiansVec(threadID + threadNum * 2),
//            contribution_count);
//
//        glm::vec3 L_i = glm::clamp(LiColor.ToGlm(), glm::vec3(0.0f), glm::vec3(10.0f));
//
//        out.push_back({
//            g.index,
//            omega_i_world.ToGlm(),
//            omega_o,
//            normal_glm,
//            L_i,
//            glm::vec3(0.0f),
//            NdotL,
//            contribution,
//            glm::vec3(0.0f)
//            });
//    }
//}
//
//
//// --------------------------------------------------------------------------
//// [SH version] collectSamplesForView  —  unchanged from main.cpp
//// --------------------------------------------------------------------------
//static void collectSamplesForView(
//    Gaussian& g,
//    const glm::vec3& omega_o,
//    float contribution,
//    MTRandom& Sampler,
//    std::vector<Gaussian>& all,
//    BVHNode* bvh,
//    std::vector<BRDFSample>& out,
//    int threadID)
//{
//    Vec3   viewVec = fromGLM(omega_o);
//    Colour SHColor = evaluateSphericalHarmonics(viewVec, g);
//    glm::vec3 sh_display = glm::clamp(SHColor.ToGlm(), glm::vec3(0.0f), glm::vec3(10.0f));
//    glm::vec3 L_o = sh_display;
//    g.testAlbedo  = sh_display;
//
//    int samplesBefore = (int)out.size();
//    monteCarloSampling(Sampler, g, all, bvh, omega_o, out, contribution, threadID);
//
//    for (int i = samplesBefore; i < (int)out.size(); i++) {
//        out[i].L_o      = L_o;
//        out[i].shColour = sh_display;
//    }
//}
//
//
//// --------------------------------------------------------------------------
//// [SG version] collectSamplesForView_SG                                 [SG]
////
//// Identical to collectSamplesForView except Lo is evaluated from the
//// Spherical Gaussians stored in allSGs[g.index] rather than from SH.
////
//// This is the key change: Lo(wo) = sum_k amp_k * exp(lambda_k*(wo.dir_k-1))
//// is a far better approximation of the true outgoing radiance for sharp
//// specular BRDFs (roughness < 0.5) than L3 SH.
//// --------------------------------------------------------------------------
//static void collectSamplesForView_SG(
//    Gaussian& g,
//    const glm::vec3& omega_o,
//    float contribution,
//    MTRandom& Sampler,
//    std::vector<Gaussian>& all,
//    BVHNode* bvh,
//    std::vector<BRDFSample>& out,
//    int threadID)
//{
//    // [SG] Evaluate Lo from Spherical Gaussians instead of SH
//    glm::vec3 L_o;
//    if ((int)allSGs.size() > g.index && !allSGs[g.index].empty())
//        L_o = evaluateSGLo(omega_o, allSGs[g.index]);
//    else
//        L_o = glm::vec3(0.5f);   // fallback if SGs not loaded
//
//    g.testAlbedo = L_o;  // for debug visualisation only
//
//    int samplesBefore = (int)out.size();
//    monteCarloSampling(Sampler, g, all, bvh, omega_o, out, contribution, threadID);
//
//    for (int i = samplesBefore; i < (int)out.size(); i++) {
//        out[i].L_o      = L_o;
//        out[i].shColour = L_o;
//    }
//}
//
//
//// --------------------------------------------------------------------------
//// collectSplatSamples_Thread  —  SH version (unchanged from main.cpp)
//// --------------------------------------------------------------------------
//void collectSplatSamples_Thread(int threadID, int startIdx, int endIdx,
//    std::vector<Gaussian>& gaussians, BVHNode* bvh)
//{
//    const int N_VIEW_SAMPLES = 64;
//    MTRandom Sampler(threadID + 1);
//
//    for (int gi = startIdx; gi < endIdx; ++gi) {
//        if (gi != 0) continue;
//
//        Gaussian& g = gaussians[gi];
//        Vec3 normal = fromGLM(g.GaussNormal).normalize();
//
//        Frame viewFrame;
//        viewFrame.fromVector(normal);
//
//        for (int v = 0; v < N_VIEW_SAMPLES; ++v) {
//            Vec3 localView = SamplingDistributions::cosineSampleHemisphere(
//                Sampler.next(), Sampler.next());
//            Vec3 omega_o_world = viewFrame.toWorld(localView);
//            if (normal.dot(omega_o_world) <= 0.0f) continue;
//
//            glm::vec3 omega_o = omega_o_world.ToGlm();
//            collectSamplesForView(g, omega_o, 1.0f,
//                Sampler, gaussians, bvh,
//                BRDFSampleList_vec[threadID], threadID);
//        }
//
//        g.testAlbedo = glm::clamp(g.color.ToGlm(), 0.02f, 0.98f);
//    }
//}
//
//
//// --------------------------------------------------------------------------
//// collectSplatSamples_Thread_SG  —  SG version                         [SG]
//// --------------------------------------------------------------------------
//void collectSplatSamples_Thread_SG(int threadID, int startIdx, int endIdx,
//    std::vector<Gaussian>& gaussians, BVHNode* bvh)
//{
//    const int N_VIEW_SAMPLES = 64;
//    MTRandom Sampler(threadID + 1);
//
//    for (int gi = startIdx; gi < endIdx; ++gi) {
//        if (gi != 0) continue;  // only splat 0 for now (front sphere point)
//
//        Gaussian& g = gaussians[gi];
//        Vec3 normal = fromGLM(g.GaussNormal).normalize();
//
//        Frame viewFrame;
//        viewFrame.fromVector(normal);
//
//        for (int v = 0; v < N_VIEW_SAMPLES; ++v) {
//            Vec3 localView = SamplingDistributions::cosineSampleHemisphere(
//                Sampler.next(), Sampler.next());
//            Vec3 omega_o_world = viewFrame.toWorld(localView);
//            if (normal.dot(omega_o_world) <= 0.0f) continue;
//
//            glm::vec3 omega_o = omega_o_world.ToGlm();
//
//            // [SG] Use SG-based Lo instead of SH
//            collectSamplesForView_SG(g, omega_o, 1.0f,
//                Sampler, gaussians, bvh,
//                BRDFSampleList_vec[threadID], threadID);
//        }
//
//        // Warm-start: DC color = base_color (same as SH version)
//        g.testAlbedo = glm::clamp(g.color.ToGlm(), 0.02f, 0.98f);
//    }
//}
//
//
//void collectSplatSamples_Thread_Indexed(int threadID, int startIdx, int endIdx,
//    std::vector<Gaussian>& gaussians, const std::vector<int>& indices, BVHNode* bvh)
//{
//    const int N_VIEW_SAMPLES = 64;
//    MTRandom Sampler(threadID + 1);
//
//    for (int gi = startIdx; gi < endIdx; ++gi) {
//        Gaussian& g = gaussians[indices[gi]];
//        Vec3 normal = fromGLM(g.GaussNormal).normalize();
//
//        Frame viewFrame;
//        viewFrame.fromVector(normal);
//
//        for (int v = 0; v < N_VIEW_SAMPLES; ++v) {
//            Vec3 localView = SamplingDistributions::cosineSampleHemisphere(
//                Sampler.next(), Sampler.next());
//            Vec3 omega_o_world = viewFrame.toWorld(localView);
//            if (normal.dot(omega_o_world) <= 0.0f) continue;
//
//            glm::vec3 omega_o = omega_o_world.ToGlm();
//            collectSamplesForView(g, omega_o, 1.0f,
//                Sampler, gaussians, bvh,
//                BRDFSampleList_vec[threadID], threadID);
//        }
//
//        g.testAlbedo = glm::clamp(g.color.ToGlm(), 0.02f, 0.98f);
//    }
//}
//
//
//// --------------------------------------------------------------------------
//// Single-threaded sample collectors
//// --------------------------------------------------------------------------
//void collectAllSplatSamples_ST(std::vector<Gaussian>& gaussians, BVHNode* bvh)
//{
//    BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(1);
//    int total = (int)gaussians.size();
//    collectSplatSamples_Thread(0, 0, total, std::ref(gaussians), bvh);
//    std::cout << "Combining BRDF samples from all threads...\n";
//    BRDFSampleList.insert(BRDFSampleList.end(),
//        BRDFSampleList_vec[0].begin(), BRDFSampleList_vec[0].end());
//}
//
//// [SG] SG-mode single-threaded sample collector                        [SG]
//void collectAllSplatSamples_ST_SG(std::vector<Gaussian>& gaussians, BVHNode* bvh)
//{
//    BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(1);
//    int total = (int)gaussians.size();
//    collectSplatSamples_Thread_SG(0, 0, total, std::ref(gaussians), bvh);
//    std::cout << "Combining BRDF samples from SG thread...\n";
//    BRDFSampleList.insert(BRDFSampleList.end(),
//        BRDFSampleList_vec[0].begin(), BRDFSampleList_vec[0].end());
//}
//
//
//void collectAllSplatSamples_MT(std::vector<Gaussian>& gaussians, BVHNode* bvh)
//{
//    BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(threadNum);
//
//    int total = (int)gaussians.size();
//    int chunkSize = (total + threadNum - 1) / threadNum;
//
//    std::vector<std::thread> threads;
//    for (int t = 0; t < threadNum; ++t) {
//        int start = t * chunkSize;
//        int end   = std::min(start + chunkSize, total);
//        if (start >= total) break;
//        threads.emplace_back(collectSplatSamples_Thread,
//            t, start, end, std::ref(gaussians), bvh);
//    }
//    for (auto& th : threads) th.join();
//
//    std::cout << "Combining BRDF samples from all threads...\n";
//    for (int t = 0; t < threadNum; ++t)
//        BRDFSampleList.insert(BRDFSampleList.end(),
//            BRDFSampleList_vec[t].begin(), BRDFSampleList_vec[t].end());
//}
//
//
//Colour BRDF(Ray& ray, std::vector<Gaussian>& in, std::vector<Gaussian>& all,
//    MTRandom& Sampler, BVHNode* bvh, int threadID = 0)
//{
//    struct Hit { float t; Gaussian* g; };
//    std::vector<Hit> hits; hits.reserve(in.size());
//
//    for (auto& g : in) {
//        float t = ray.dir.dot((g.pos - ray.o));
//        if (t <= 0) continue;
//        hits.push_back({ t, &g });
//    }
//    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
//        return a.t < b.t; });
//
//    Colour color(0, 0, 0);
//    float tr = 1.f;
//
//    for (auto& h : hits) {
//        Gaussian& g = *h.g;
//        float alpha = g.computeAlpha(ray);
//        if (tr < 0.001f) break;
//        Vec3   viewDir = (ray.o - g.pos).normalize();
//        Colour SHColor = evaluateSphericalHarmonics(viewDir, g);
//        color = color + (SHColor * alpha * tr);
//        tr    *= (1.0f - alpha);
//    }
//    color.correct();
//    return color;
//}
//
//
//void setCamera(Camera& camera, RTCamera& viewCamera) {
//    Vec3 from(0.0f, 0.0f, -5.0f);
//    Vec3 to(0.0f, 0.0f, 0.0f);
//    Vec3 up(0.0f, -1.0f, 0.0f);
//    viewCamera.from = from;
//    viewCamera.to   = to;
//    viewCamera.up   = up;
//    viewCamera.setCamera(&camera);
//}
//
//
//// --------------------------------------------------------------------------
//// loadSphereSamples — forward declaration  (defined in SpherePointOptim.h)
//// --------------------------------------------------------------------------
//// Already declared in SpherePointOptim.h, no redeclaration needed here.
//
//
//// ==========================================================================
////  Test functions
//// ==========================================================================
//
//// --------------------------------------------------------------------------
//// spherePointTestSG
////
//// SG-mode BRDF recovery.  Pipeline:
////   1. python SphereSGGen.py     ->  sphere_sg_scene.ply
////   2. Compile main2.cpp
////   3. Run this executable       ->  recover BRDF from splat 0
////
//// Key difference from spherePointTestPLY (main.cpp):
////   Lo(wo) is evaluated from K_SG Spherical Gaussian lobes stored in the PLY,
////   not from L3 SH.  This eliminates the ~4-5% base_color error caused by
////   the SH spectral limitation (L3 SH cannot represent a sharp specular lobe).
////
//// Expected recovery (ground truth):
////   base_color = (0.8, 0.2, 0.1)
////   metallic   = 0.0
////   roughness  = 0.3
////   specular   = 0.5
//// --------------------------------------------------------------------------
//void spherePointTestSG()
//{
//    std::cout << "\n";
//    std::cout << "==========================================================\n";
//    std::cout << "  Sphere Point BRDF Recovery  (SG mode)\n";
//    std::cout << "==========================================================\n\n";
//
//    // --- Step 1: Load PLY with SG lobes ---
//    std::cout << "  Step 1: Loading sphere_sg_scene.ply (with SG lobes)...\n";
//    std::cout << "          (generate with: python SphereSGGen.py)\n\n";
//
//    std::vector<Gaussian> gaussians;
//    parsePLYWithSG("sphere_sg_scene.ply", gaussians, K_SG_DEFAULT);
//
//    if (gaussians.empty()) {
//        std::cerr << "  ERROR: No Gaussians loaded. Run SphereSGGen.py first.\n";
//        return;
//    }
//
//    std::cout << "\n  Loaded " << gaussians.size()
//              << " Gaussians (splat 0 = front sphere point, N=[0,0,1])\n";
//
//    // Print DC warm-start colour for splat 0
//    glm::vec3 warmStart = glm::clamp(gaussians[0].color.ToGlm(), 0.02f, 0.98f);
//    std::cout << "  Warm-start base_color from PLY DC: ("
//              << warmStart.r << ", " << warmStart.g << ", " << warmStart.b << ")\n";
//
//    // Print a sample SG evaluation for splat 0 at wo=N to sanity-check the load
//    if (!allSGs.empty() && !allSGs[0].empty()) {
//        glm::vec3 N(0.f, 0.f, 1.f);
//        glm::vec3 Lo_N = evaluateSGLo(N, allSGs[0]);
//        std::cout << "  SG Lo at wo=N=[0,0,1]:  ("
//                  << Lo_N.r << ", " << Lo_N.g << ", " << Lo_N.b << ")\n";
//        std::cout << "  (expected near GT Lo(N) computed by SphereSGGen.py)\n\n";
//    }
//
//    // Set warm-start from DC for all splats
//    for (auto& g : gaussians)
//        g.testAlbedo = glm::clamp(g.color.ToGlm(), 0.02f, 0.98f);
//
//    // --- Step 2: Build BVH ---
//    std::cout << "  Step 2: Building BVH...\n";
//    BVHNode bvh;
//    bvh.build(gaussians);
//
//    // --- Step 3: Collect samples (SG Lo used for each view direction) ---
//    std::cout << "  Step 3: Collecting hemisphere samples for splat 0 (SG Lo)...\n";
//    collectAllSplatSamples_ST_SG(gaussians, &bvh);
//    std::cout << "  Collected " << BRDFSampleList.size() << " samples\n\n";
//
//    // Optional: dump samples for inspection
//    writeBRDFSamples("BRDF_Samples_SG.csv", gaussians);
//
//    // --- Step 4: Optimise Disney BRDF ---
//    std::cout << "  Step 4: Optimising Disney BRDF (3000 iterations)...\n\n";
//    std::cout << "  Ground truth:  bc=(0.8, 0.2, 0.1)  met=0.0  rough=0.3  spec=0.5\n";
//    std::cout << "  SH mode gave:  bc~(0.763, ?, ?)  (L3 SH spectral limit)\n";
//    std::cout << "  SG mode target: bc close to (0.8, 0.2, 0.1)\n\n";
//
//    optimizeDisneyBRDFAutodiff(BRDFSampleList, gaussians, /*maxIter=*/3000);
//}
//
//
//// --------------------------------------------------------------------------
//// spherePointTestPLY  (original SH-based BVH mode -- kept for comparison)
//// --------------------------------------------------------------------------
//void spherePointTestPLY()
//{
//    std::cout << "=== Sphere Point BRDF Recovery Test (PLY / SH mode) ===\n\n";
//
//    std::vector<Gaussian> gaussians;
//    parsePLY("sphere_scene.ply", gaussians, "sphere_scene.ply");
//    if (gaussians.empty()) {
//        std::cerr << "  ERROR: No Gaussians loaded.\n"; return;
//    }
//    std::cout << "  Loaded " << gaussians.size() << " Gaussians\n\n";
//
//    BVHNode bvh;
//    bvh.build(gaussians);
//
//    collectAllSplatSamples_ST(gaussians, &bvh);
//    std::cout << "  Collected " << BRDFSampleList.size() << " samples\n\n";
//
//    writeBRDFSamples("BRDF_Samples_SH.csv", gaussians);
//    optimizeDisneyBRDFAutodiff(BRDFSampleList, gaussians, /*maxIter=*/1000);
//}
//
//
//// --------------------------------------------------------------------------
//// spherePointTestCSV  (CSV mode -- Loss(GT)=0 guaranteed)
//// --------------------------------------------------------------------------
//void spherePointTestCSV()
//{
//    std::cout << "=== Sphere Point BRDF Recovery Test (CSV mode -- Loss(GT)=0) ===\n\n";
//
//    std::vector<Gaussian> gaussians;
//    parsePLY("sphere_scene.ply", gaussians, "sphere_scene.ply");
//    glm::vec3 warmStart(0.5f);
//    if (!gaussians.empty()) {
//        warmStart = glm::clamp(
//            glm::vec3(gaussians[0].ZeroSH.x * SH_C0 + 0.5f,
//                       gaussians[0].ZeroSH.y * SH_C0 + 0.5f,
//                       gaussians[0].ZeroSH.z * SH_C0 + 0.5f),
//            0.02f, 0.98f);
//        std::cout << "  Warm-start bc from PLY DC: ("
//                   << warmStart.r << ", " << warmStart.g << ", " << warmStart.b << ")\n\n";
//    }
//
//    std::cout << "  Loading sphere_samples.csv ...\n";
//    std::vector<BRDFSample> samples = loadSphereSamples("sphere_samples.csv");
//    if (samples.empty()) {
//        std::cerr << "\n  ERROR: No samples. Run SphereSampleGenFromPLY.py first.\n";
//        return;
//    }
//
//    for (auto& g : gaussians)
//        g.testAlbedo = warmStart;
//
//    std::cout << "\n  Optimising Disney BRDF (CSV mode, Loss(GT)=0)...\n";
//    optimizeDisneyBRDFAutodiff(samples, gaussians, /*maxIter=*/3000);
//}
//
//
//// --------------------------------------------------------------------------
//// main
//// --------------------------------------------------------------------------
//int main()
//{
//    // -----------------------------------------------------------------------
//    // SG mode:  uses Spherical Gaussians for Lo -- best BC convergence from PLY
//    //   1. python SphereSGGen.py          ->  sphere_sg_scene.ply
//    //   2. Compile main2.cpp
//    //   3. Run executable
//    spherePointTestSG();
//
//    // -----------------------------------------------------------------------
//    // CSV mode:  Loss(GT)=0 guaranteed -- bc/rough/spec converge to exact GT
//    //   1. python SpherePLYGen.py          ->  sphere_scene.ply
//    //   2. python SphereSampleGenFromPLY.py ->  sphere_samples.csv
//    //   3. Compile main2.cpp
//    //   4. Run executable (comment out spherePointTestSG above, uncomment below)
//    //spherePointTestCSV();
//
//    // -----------------------------------------------------------------------
//    // SH mode:  original BVH/SH pipeline -- bc converges to ~0.763
//    //           (L3 SH spectral limit for roughness=0.3)
//    //spherePointTestPLY();
//
//    return 0;
//}
