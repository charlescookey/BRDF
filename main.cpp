#include <iostream>
#include <fstream>
#include <set>
#include <thread>

#include "happly.h"
#include "GamesEngineeringBase.h"

#include "Math.h"
#include "Sampling.h"
#include "Imaging.h"

#include "BRDFOptimizerSamples.h"
#include "SingleOptim.h"
#include "DisneyBRDFOptimizerSimple.h"
#include "CosBRDF_Disney.h"

#include "BRDF_Optim_AutoDiff.h"
//#include "BRDF_Optim_Progressive.h"

//#include "SphereScene.h"
//#include "SphereOptim.h"
#include "SpherePointOptim.h"

#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0  1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2  0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4  0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1  2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3  0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5  1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

#define tileSize 16


std::vector<BRDFSample> BRDFSampleList;
std::vector<std::vector<BRDFSample>> BRDFSampleList_vec;


void writeBRDFSamples(const std::string& filename, std::vector<Gaussian>& gaussians) {
	std::ofstream file(filename);
	if (!file.is_open()) return;

	file << "splatIndex,omega_i.x,omega_i.y,omega_i.z,omega_o.x,omega_o.y,omega_o.z,normal.x,normal.y,normal.z,L_i.x,L_i.y,L_i.z,L_o.x,L_o.y,L_o.z,cosTheta,weight,SH.r, SH.g, SH.b, albedo.r,albedo.g,albedo.b\n";

	for (const auto& s : BRDFSampleList) {
		file << s.splatIndex << ","
			<< s.omega_i.x << "," << s.omega_i.y << "," << s.omega_i.z << ","
			<< s.omega_o.x << "," << s.omega_o.y << "," << s.omega_o.z << ","
			<< s.normal.x << "," << s.normal.y << "," << s.normal.z << ","
			<< s.L_i.x << "," << s.L_i.y << "," << s.L_i.z << ","
			<< s.L_o.x << "," << s.L_o.y << "," << s.L_o.z << ","
			<< s.cosTheta << ","
			<< s.weight << ","
			<< s.shColour.x << "," << s.shColour.y << "," << s.shColour.z << ","
			<< gaussians[s.splatIndex].color.r << ","
			<< gaussians[s.splatIndex].color.g << ","
			<< gaussians[s.splatIndex].color.b
			<< "\n";
	}

	file << "\n\n";
	file.close();
	std::cout << "Saved " << BRDFSampleList.size()
		<< " BRDF samples to " << filename << std::endl;
}


Colour evaluateSphericalHarmonics(const Vec3& viewDir, Gaussian& gaussian) {
	Vec3 dir = viewDir.normalize();
	float x = dir.x, y = dir.y, z = dir.z;
	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, xz = x * z, yz = y * z;

	Vec3 color = gaussian.ZeroSH * SH_C0;

	color = color
		- (Vec3(gaussian.higherSH[0], gaussian.higherSH[1], gaussian.higherSH[2]) * SH_C1 * y)
		+ (Vec3(gaussian.higherSH[3], gaussian.higherSH[4], gaussian.higherSH[5]) * SH_C1 * z)
		- (Vec3(gaussian.higherSH[6], gaussian.higherSH[7], gaussian.higherSH[8]) * SH_C1 * x);

	color = color
		+ (Vec3(gaussian.higherSH[9], gaussian.higherSH[10], gaussian.higherSH[11]) * SH_C2_0 * xy)
		+ (Vec3(gaussian.higherSH[12], gaussian.higherSH[13], gaussian.higherSH[14]) * SH_C2_1 * yz)
		+ (Vec3(gaussian.higherSH[15], gaussian.higherSH[16], gaussian.higherSH[17]) * SH_C2_2 * (2.0f * zz - xx - yy))
		+ (Vec3(gaussian.higherSH[18], gaussian.higherSH[19], gaussian.higherSH[20]) * SH_C2_3 * xz)
		+ (Vec3(gaussian.higherSH[21], gaussian.higherSH[22], gaussian.higherSH[23]) * SH_C2_4 * (xx - yy));

	color = color
		+ (Vec3(gaussian.higherSH[24], gaussian.higherSH[25], gaussian.higherSH[26]) * SH_C3_0 * y * (3.0f * xx - yy))
		+ (Vec3(gaussian.higherSH[27], gaussian.higherSH[28], gaussian.higherSH[29]) * SH_C3_1 * xy * z)
		+ (Vec3(gaussian.higherSH[30], gaussian.higherSH[31], gaussian.higherSH[32]) * SH_C3_2 * y * (4.0f * zz - xx - yy))
		+ (Vec3(gaussian.higherSH[33], gaussian.higherSH[34], gaussian.higherSH[35]) * SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy))
		+ (Vec3(gaussian.higherSH[36], gaussian.higherSH[37], gaussian.higherSH[38]) * SH_C3_4 * x * (4.0f * zz - xx - yy))
		+ (Vec3(gaussian.higherSH[39], gaussian.higherSH[40], gaussian.higherSH[41]) * SH_C3_5 * z * (xx - yy))
		+ (Vec3(gaussian.higherSH[42], gaussian.higherSH[43], gaussian.higherSH[44]) * SH_C3_6 * x * (xx - 3.0f * yy));

	Colour c;
	c += color;
	c += Vec3(0.5f);
	return c;
}

Colour viewIndependent(Gaussian& gaussian) {
	Vec3 color = gaussian.ZeroSH * SH_C0;
	Colour c;
	c += color;
	c += Vec3(0.5f);
	return c;
}


void printNormals(std::string filename) {
	happly::PLYData plyInNorm(filename.c_str());
	std::vector<double> nx = plyInNorm.getElement("vertex").getProperty<double>("nx");
	std::vector<double> ny = plyInNorm.getElement("vertex").getProperty<double>("ny");
	std::vector<double> nz = plyInNorm.getElement("vertex").getProperty<double>("nz");

	std::ofstream outfile("normals.txt");
	for (size_t i = 0; i < nx.size(); i++)
		outfile << i << " " << nx[i] << " " << ny[i] << " " << nz[i] << "\n";
	outfile.close();
}


void parsePLY(std::string filename, std::vector<Gaussian>& gaussians, std::string Normalfilename) {
	happly::PLYData plyIn(filename.c_str());
	std::vector<float> prop1 = plyIn.getElement("vertex").getProperty<float>("x");
	std::vector<float> prop2 = plyIn.getElement("vertex").getProperty<float>("y");
	std::vector<float> prop3 = plyIn.getElement("vertex").getProperty<float>("z");

	int size = prop1.size();
	gaussians = std::vector<Gaussian>(size);

	for (size_t i = 0; i < prop1.size(); i++) {
		gaussians[i].pos = Vec3(prop1[i], prop2[i], prop3[i]);
		gaussians[i].index = i;
	}

	happly::PLYData plyInNorm(Normalfilename.c_str());
	std::vector<double> nx = plyInNorm.getElement("vertex").getProperty<double>("nx");
	std::vector<double> ny = plyInNorm.getElement("vertex").getProperty<double>("ny");
	std::vector<double> nz = plyInNorm.getElement("vertex").getProperty<double>("nz");

	for (size_t i = 0; i < prop1.size(); i++)
		gaussians[i].GaussNormal = glm::vec3(nx[i], ny[i], nz[i]);

	prop1 = plyIn.getElement("vertex").getProperty<float>("f_dc_0");
	prop2 = plyIn.getElement("vertex").getProperty<float>("f_dc_1");
	prop3 = plyIn.getElement("vertex").getProperty<float>("f_dc_2");

	for (size_t i = 0; i < prop1.size(); i++) {
		gaussians[i].ZeroSH = Vec3(prop1[i], prop2[i], prop3[i]);
		gaussians[i].color = viewIndependent(gaussians[i]);
		gaussians[i].testColor = gaussians[i].color.ToGlm();
	}

	prop1 = plyIn.getElement("vertex").getProperty<float>("scale_0");
	prop2 = plyIn.getElement("vertex").getProperty<float>("scale_1");
	prop3 = plyIn.getElement("vertex").getProperty<float>("scale_2");

	for (size_t i = 0; i < prop1.size(); i++) {
		gaussians[i].scale = Vec3(prop1[i], prop2[i], prop3[i]);
		gaussians[i].compute_gaussian_aabb();
	}

	std::vector<float> prop4 = plyIn.getElement("vertex").getProperty<float>("opacity");
	for (size_t i = 0; i < prop1.size(); i++)
		gaussians[i].opacity = sigmoid(prop4[i]);

	prop1 = plyIn.getElement("vertex").getProperty<float>("rot_0");
	prop2 = plyIn.getElement("vertex").getProperty<float>("rot_1");
	prop3 = plyIn.getElement("vertex").getProperty<float>("rot_2");
	prop4 = plyIn.getElement("vertex").getProperty<float>("rot_3");

	for (size_t i = 0; i < prop1.size(); i++) {
		gaussians[i].rotation = Vec3(prop1[i], prop2[i], prop3[i], prop4[i]);
		gaussians[i].compute_gaussian_covariance();
	}

	for (int i = 0; i < 45; i++) {
		std::string name = "f_rest_" + std::to_string(i);
		prop1 = plyIn.getElement("vertex").getProperty<float>(name);
		for (size_t j = 0; j < prop1.size(); j++)
			gaussians[j].higherSH.push_back(prop1[j]);
	}
}

Colour GaussianColor(Ray& ray, std::vector<Gaussian>& in, int& contribution_count)
{
	struct Hit { float t; Gaussian* g; };
	std::vector<Hit> hits; hits.reserve(in.size());
	contribution_count = 0;

	for (auto& g : in) {
		float t = ray.dir.dot((g.pos - ray.o));
		if (t <= 0) continue;
		hits.push_back({ t, &g });
	}
	std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
		return a.t < b.t;
		});

	Colour color(0, 0, 0);
	float tr = 1.f;

	for (auto& h : hits) {
		Gaussian& g = *h.g;
		float alpha = g.computeAlpha(ray);

		if (tr < 0.001f) break;

		Vec3 viewDir = (ray.o - g.pos).normalize();
		Colour SHColor = evaluateSphericalHarmonics(viewDir, g);

		float contribution = alpha * tr;
		if (contribution > 0.05f) {

			//std::cout << "Gassian " << g.index << " with a contribution of " << contribution <<" with color " << SHColor.r << " , " << SHColor.g << " , " << SHColor.b << "\n";
			color = color + (SHColor * alpha * tr);
			tr *= (1.0f - alpha);
			contribution_count++;
		}
	}
	//std::cout << "Final color: " << color.r << " , " << color.g << " , " << color.b << "\n";
	//if (correct) color.correct();
	return color;
}

Colour GaussianNormal(Ray& ray, std::vector<Gaussian>& in, int& contribution_count)
{
	struct Hit { float t; Gaussian* g; };
	std::vector<Hit> hits; hits.reserve(in.size());
	contribution_count = 0;

	for (auto& g : in) {
		float t = ray.dir.dot((g.pos - ray.o));
		if (t <= 0) continue;
		hits.push_back({ t, &g });
	}
	std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
		return a.t < b.t;
		});

	Colour color(0, 0, 0);
	float tr = 1.f;

	for (auto& h : hits) {
		Gaussian& g = *h.g;
		float alpha = g.computeAlpha(ray);

		if (tr < 0.001f) break;

		Colour SHColor = Colour(g.GaussNormal.x, g.GaussNormal.y, g.GaussNormal.z);

		float contribution = alpha * tr;
		if (contribution > 0.05f) {
			color = color + (SHColor * alpha * tr);
			tr *= (1.0f - alpha);
			contribution_count++;
		}
	}
	return color;
}

Colour GaussianNormal_Single(Ray& ray, std::vector<Gaussian>& in, int& contribution_count)
{
	struct Hit { float t; Gaussian* g; };
	std::vector<Hit> hits; hits.reserve(in.size());
	contribution_count = 0;

	for (auto& g : in) {
		float t = ray.dir.dot((g.pos - ray.o));
		if (t <= 0) continue;
		hits.push_back({ t, &g });
	}
	std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
		return a.t < b.t;
		});

	Colour color(0, 0, 0);
	float tr = 1.f;
	//find the gaussian with the highest contribution and return its normal
	float max_contribution = 0.f;
	Gaussian* max_gaussian = nullptr;
	for (auto& h : hits) {
		Gaussian& g = *h.g;
		float alpha = g.computeAlpha(ray);

		if (tr < 0.001f) break;

		float contribution = alpha * tr;
		if (contribution > max_contribution) {
			max_contribution = contribution;
			max_gaussian = &g;
		}
	}
	if (max_gaussian) {
		return Colour(max_gaussian->GaussNormal.x, max_gaussian->GaussNormal.y, max_gaussian->GaussNormal.z);
	}
	return color;
}

Colour GaussianAlbedo(Ray& ray, std::vector<Gaussian>& in)
{
	struct Hit { float t; Gaussian* g; };
	std::vector<Hit> hits; hits.reserve(in.size());

	for (auto& g : in) {
		float t = ray.dir.dot((g.pos - ray.o));
		if (t <= 0) continue;
		hits.push_back({ t, &g });
	}
	std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
		return a.t < b.t;
		});

	Colour color(0, 0, 0);
	float tr = 1.f;

	for (auto& h : hits) {
		Gaussian& g = *h.g;
		float alpha = g.computeAlpha(ray);
		if (tr < 0.001f) break;
		Colour SHColor = fromGLMC(g.testAlbedo);
		color = color + (SHColor * alpha * tr);
		tr *= (1.0f - alpha);
	}
	color.correct();
	return color;
}


// -------------------------------------------------------------------------
// monteCarloSampling  (FIXED)
//
// Returns one BRDFSample per hemisphere sample (not one per splat).
// Each sample contains:
//   omega_i  : normalized world-space direction (from Frame.toWorld)
//   L_i      : raw per-sample radiance, clamped to [0, 2]
//   NdotL    : cosine of the angle between normal and omega_i
//
// Changes from old version:
//   - Normal is flipped toward the VIEW direction (omega_o), not toward
//     the world origin. The old check `normal.dot(g.pos) < 0` was wrong.
//   - Each sample is stored individually -- no averaging of omega_i.
//   - L_i is raw radiance per sample, NOT the integrated MC estimate.
//     The optimizer owns the integration (pi/N * sum).
//   - cosTheta weighting is NOT applied here -- that belongs in the
//     optimizer's rendering equation sum.
// -------------------------------------------------------------------------
void monteCarloSampling(
	MTRandom& Sampler,
	Gaussian& g,
	std::vector<Gaussian>& all,
	BVHNode* bvh,
	const glm::vec3& omega_o,        // view direction, used for normal flip
	std::vector<BRDFSample>& out,    // one entry appended per valid sample
	float contribution,              // alpha*tr weight, passed through to sample
	int threadID = 0)
{
	const int N_SAMPLES = 128;

	Vec3 normal = fromGLM(g.GaussNormal);

	// Flip normal toward the camera (view direction), not toward world origin.
	// omega_o points FROM the splat TO the camera.
	if (normal.dot(fromGLM(omega_o)) < 0.0f)
		normal = normal * -1.f;

	glm::vec3 normal_glm = normal.ToGlm();
	Frame frame;
	frame.fromVector(normal);

	for (int s = 0; s < N_SAMPLES; ++s) {
		// Sample in local hemisphere (z > 0 guaranteed by cosine sampling)
		Vec3 localDir = SamplingDistributions::cosineSampleHemisphere(
			Sampler.next(), Sampler.next());

		// Transform to world space
		Vec3 omega_i_world = frame.toWorld(localDir);

		float NdotL = normal.dot(omega_i_world);
		if (NdotL <= 0.0f) continue; // safety check, should rarely trigger

		// Shoot secondary ray
		Ray newRay;
		newRay.init(g.pos + (omega_i_world * EPSILON), omega_i_world);
		bvh->traverse(newRay, all, threadID + threadNum * 2);

		int contribution_count = 0;
		Colour LiColor = GaussianColor(
			newRay,
			bvh->getIntersectedGaussiansVec(threadID + threadNum * 2),
			contribution_count);  // false = no tonemapping -- we want raw radiance

		// Clamp to [0, 2]: kills negative SH ringing, keeps physical range
		glm::vec3 L_i = glm::clamp(LiColor.ToGlm(), glm::vec3(0.0f), glm::vec3(10.0f));

		// Store one sample per hemisphere direction.
		// omega_i is the individual sample direction (normalized).
		// L_i is the raw radiance in that direction.
		// NdotL is stored so the optimizer can use it in the rendering equation.
		// L_o is filled in by the caller (it's a property of the splat, not the sample).
		out.push_back({
			g.index,
			omega_i_world.ToGlm(),   // omega_i: normalized world-space direction
			omega_o,                 // omega_o: view direction (same for all samples of this splat/ray)
			normal_glm,           // surface normal (stored as-is for the optimizer)
			L_i,                     // raw incoming radiance along omega_i
			glm::vec3(0.0f),         // L_o: filled in below by the caller
			NdotL,                   // cosTheta = N dot omega_i
			contribution,            // alpha * tr weight
			glm::vec3(0.0f)          // shColour: filled in below by the caller
			});
	}
}


// -------------------------------------------------------------------------
// Helper: given a splat and ONE view direction omega_o, evaluate SH to get
// L_o (with Reinhard inversion), shoot N_SAMPLES hemisphere rays for L_i,
// and push all resulting BRDFSamples into `out`.
// Called once for the real camera ray and N_VIEW_SAMPLES extra times with
// randomly sampled view directions so the optimizer sees angular variation.
// -------------------------------------------------------------------------
static void collectSamplesForView(
	Gaussian& g,
	const glm::vec3& omega_o,   // view direction (splat -> camera), normalized
	float contribution,
	MTRandom& Sampler,
	std::vector<Gaussian>& all,
	BVHNode* bvh,
	std::vector<BRDFSample>& out,
	int threadID)
{
	// --- L_o: SH evaluated in this view direction ---
	// evaluateSphericalHarmonics returns SH_eval + 0.5, which is the raw
	// linear colour directly. GenMiniSplat.py stores (base_color - 0.5)/SH_C0
	// in the DC coefficient so that SH_C0 * coeff + 0.5 = base_color.
	// No Reinhard is baked in -- do NOT invert anything here.
	// L_i from GaussianColor is also raw linear, so both are in the same space.
	Vec3   viewVec = fromGLM(omega_o);
	Colour SHColor = evaluateSphericalHarmonics(viewVec, g);
	glm::vec3 sh_display = glm::clamp(SHColor.ToGlm(), glm::vec3(0.0f), glm::vec3(10.0f));
	glm::vec3 L_o = sh_display;
	g.testAlbedo = sh_display;  // for debug visualization only -- not used in optimizer

	// --- hemisphere samples for L_i ---
	int samplesBefore = (int)out.size();
	monteCarloSampling(Sampler, g, all, bvh, omega_o, out, contribution, threadID);

	// Fill in the shared-per-view fields for every sample just added
	for (int i = samplesBefore; i < (int)out.size(); i++) {
		out[i].L_o = L_o;
		out[i].shColour = sh_display;
	}
}


// BRDF is now rendering-only -- no sample collection here.
Colour BRDF(Ray& ray, std::vector<Gaussian>& in, std::vector<Gaussian>& all,
	MTRandom& Sampler, BVHNode* bvh, int threadID = 0)
{
	struct Hit { float t; Gaussian* g; };
	std::vector<Hit> hits; hits.reserve(in.size());

	for (auto& g : in) {
		float t = ray.dir.dot((g.pos - ray.o));
		if (t <= 0) continue;
		hits.push_back({ t, &g });
	}
	std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
		return a.t < b.t;
		});

	Colour color(0, 0, 0);
	float tr = 1.f;

	for (auto& h : hits) {
		Gaussian& g = *h.g;
		float alpha = g.computeAlpha(ray);
		if (tr < 0.001f) break;
		Vec3   viewDir = (ray.o - g.pos).normalize();
		Colour SHColor = evaluateSphericalHarmonics(viewDir, g);
		color = color + (SHColor * alpha * tr);
		tr *= (1.0f - alpha);
	}
	color.correct();
	return color;
}


// -------------------------------------------------------------------------
// collectSplatSamples_Thread
//
// Each thread owns a contiguous slice of the gaussians array.
// For every splat in the slice it samples N_VIEW_SAMPLES view directions
// from the upper hemisphere of the splat normal, evaluates SH to get L_o,
// and shoots N_HEMI_SAMPLES hemisphere rays to measure L_i.
//
// No camera ray tracing, no tile dispatch, no redundant splat hits.
// Each splat is visited exactly once total across all threads.
// -------------------------------------------------------------------------
void collectSplatSamples_Thread( int threadID, int startIdx, int endIdx, std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	// View directions per splat. 16 gives good angular coverage for
	// roughness recovery; reduce to 8 if performance is a concern.
	const int N_VIEW_SAMPLES = 64;

	MTRandom Sampler(threadID + 1); // distinct seed per thread

	for (int gi = startIdx; gi < endIdx; ++gi) {
		//temporarily, only collect for first splat
		//if (gi != 0) continue;

		Gaussian& g = gaussians[gi];

		Vec3 normal = fromGLM(g.GaussNormal);
		// Normalise in case the stored normal isn't unit length
		normal = normal.normalize();

		Frame viewFrame;
		viewFrame.fromVector(normal);

		for (int v = 0; v < N_VIEW_SAMPLES; ++v) {
			// Sample a view direction from the upper hemisphere of the normal.
			// Cosine-weighted gives denser sampling near the normal where the
			// BRDF response is strongest.
			Vec3 localView = SamplingDistributions::cosineSampleHemisphere(
				Sampler.next(), Sampler.next());
			Vec3 omega_o_world = viewFrame.toWorld(localView);

			// Safety: discard if it ended up below the normal (rounding)
			if (normal.dot(omega_o_world) <= 0.0f) continue;

			glm::vec3 omega_o = omega_o_world.ToGlm();

			collectSamplesForView(g, omega_o, 1.0f,
				Sampler, gaussians, bvh,
				BRDFSampleList_vec[threadID], threadID);
		}

		// Fix warm-start: reset to DC color after the view loop.
		// collectSamplesForView sets g.testAlbedo = SH(last_wo) on every call,
		// leaving it at a random Lo value instead of the intended base_color.
		// g.color = viewIndependent(g) = ZeroSH * SH_C0 + 0.5 = base_color.
		g.testAlbedo = glm::clamp(g.color.ToGlm(), 0.02f, 0.98f);
	}
}

static bool worldToScreen(Vec3 W,
                           Vec3 camFrom, Vec3 camTo, Vec3 camUp, float fovDeg,
                           int imgW, int imgH,
                           int& outX, int& outY)
{
    Vec3 forward = (camTo - camFrom).normalize();
    Vec3 right   = forward.cross(camUp).normalize();
    Vec3 upCam   = right.cross(forward);    // true orthogonal up in camera space

    Vec3  d  = W - camFrom;
    float cx = d.dot(right);
    float cy = d.dot(upCam);
    float cz = d.dot(forward);

    if (cz <= 0.001f) return false;         // behind camera

    float f = 1.0f / std::tan(fovDeg * 0.5f * (float)M_PI / 180.0f);

    float ndc_x =  cx * f / cz;
    float ndc_y =  cy * f / cz;

    // Screen: x left→right, y top→bottom  (flip ndc_y)
    outX = (int)((ndc_x + 1.0f) * 0.5f * (float)imgW);
    outY = (int)((1.0f - ndc_y) * 0.5f * (float)imgH);

    return outX >= 0 && outX < imgW && outY >= 0 && outY < imgH;
}

// Draw a 1-pixel-wide line using Bresenham's algorithm.
// r, g, b are in [0, 255].
// Bresenham line on canvas (r,g,b in 0-255)
static void drawLine(GamesEngineeringBase::Window* canvas,
	int W, int H,
	int x0, int y0, int x1, int y1,
	unsigned char r, unsigned char g, unsigned char b)
{
	int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
	int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
	int err = dx + dy;
	while (true) {
		if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H)
			canvas->draw(x0, y0, (float)r, (float)g, (float)b);
		if (x0 == x1 && y0 == y1) break;
		int e2 = 2 * err;
		if (e2 >= dy) { err += dy; x0 += sx; }
		if (e2 <= dx) { err += dx; y0 += sy; }
	}
}

// Draw a vector of Rays as line segments on the canvas.
// Each ray is drawn from its origin to origin + dir * rayLen.
// Uses camera.projectOntoCamera() directly — no custom camera math needed.
void drawRays(const std::vector<Ray>& rays,
	Camera& camera,
	GamesEngineeringBase::Window* canvas,
	float rayLen = 1.0f,
	unsigned char r = 0, unsigned char g = 255, unsigned char b = 255)
{
	int W = (int)camera.width;
	int H = (int)camera.height;

	for (const Ray& ray : rays) {
		Vec3 tip = ray.o + ray.dir * rayLen;

		float sx, sy, ex, ey;
		bool originVis = camera.projectOntoCamera(ray.o, sx, sy);
		bool tipVis = camera.projectOntoCamera(tip, ex, ey);

		if (!originVis && !tipVis) continue;

		// Clamp to image border so partial rays still draw
		auto clampI = [](float v, int lo, int hi) {
			return std::max(lo, std::min(hi, (int)v));
			};
		drawLine(canvas, W, H,
			clampI(sx, 0, W - 1), clampI(sy, 0, H - 1),
			clampI(ex, 0, W - 1), clampI(ey, 0, H - 1),
			r, g, b);
	}
}

std::vector<Ray> rays;

void monteRays(
	MTRandom& Sampler,
	Gaussian& g,
	std::vector<Gaussian>& all,
	BVHNode* bvh,
	const glm::vec3& omega_o,        // view direction, used for normal flip
	std::vector<BRDFSample>& out,    // one entry appended per valid sample
	float contribution,              // alpha*tr weight, passed through to sample
	int threadID = 0)
{
	const int N_SAMPLES = 10;

	Vec3 normal = fromGLM(g.GaussNormal);

	// Flip normal toward the camera (view direction), not toward world origin.
	// omega_o points FROM the splat TO the camera.
	if (normal.dot(fromGLM(omega_o)) < 0.0f)
		normal = normal * -1.f;

	glm::vec3 normal_glm = normal.ToGlm();
	Frame frame;
	frame.fromVector(normal);

	for (int s = 0; s < N_SAMPLES; ++s) {
		// Sample in local hemisphere (z > 0 guaranteed by cosine sampling)
		Vec3 localDir = SamplingDistributions::cosineSampleHemisphere(
			Sampler.next(), Sampler.next());

		// Transform to world space
		Vec3 omega_i_world = frame.toWorld(localDir);

		float NdotL = normal.dot(omega_i_world);
		if (NdotL <= 0.0f) continue; // safety check, should rarely trigger

		// Shoot secondary ray
		Ray newRay;
		newRay.init(g.pos + (omega_i_world * EPSILON), omega_i_world);
		rays.push_back(newRay);
		
	}
}

void getRays(int threadID, int startIdx, int endIdx, std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	const int N_VIEW_SAMPLES = 6;

	MTRandom Sampler(threadID + 1); // distinct seed per thread

	for (int gi = startIdx; gi < endIdx; ++gi) {
		Gaussian& g = gaussians[gi];

		Vec3 normal = fromGLM(g.GaussNormal);
		std::cout << "Normal: " << normal.x << " , " << normal.y << " , " << normal.z << "\n";
		normal = normal.normalize();

		Frame viewFrame;
		viewFrame.fromVector(normal);

		Ray normalRay(g.pos, normal);
		//rays.push_back(normalRay);
		//break;

		for (int v = 0; v < N_VIEW_SAMPLES; ++v) {

			Vec3 localView = SamplingDistributions::cosineSampleHemisphere(
				Sampler.next(), Sampler.next());
			Vec3 omega_o_world = viewFrame.toWorld(localView);

			if (normal.dot(omega_o_world) <= 0.0f) continue;

			glm::vec3 omega_o = omega_o_world.ToGlm();

			monteRays(Sampler, g, gaussians, bvh, omega_o, BRDFSampleList_vec[threadID], 1.0f, threadID);
		}
		g.testAlbedo = glm::clamp(g.color.ToGlm(), 0.02f, 0.98f);
	}
}

void collectSplatSamples_Thread_Indexed(int threadID, int startIdx, int endIdx, std::vector<Gaussian>& gaussians, const std::vector<int>& indices, BVHNode* bvh)
{
	// View directions per splat. 16 gives good angular coverage for
	// roughness recovery; reduce to 8 if performance is a concern.
	const int N_VIEW_SAMPLES = 64;

	MTRandom Sampler(threadID + 1); // distinct seed per thread

	for (int gi = startIdx; gi < endIdx; ++gi) {
		Gaussian& g = gaussians[indices[gi]];

		Vec3 normal = fromGLM(g.GaussNormal);
		// Normalise in case the stored normal isn't unit length
		normal = normal.normalize();

		Frame viewFrame;
		viewFrame.fromVector(normal);

		for (int v = 0; v < N_VIEW_SAMPLES; ++v) {
			// Sample a view direction from the upper hemisphere of the normal.
			// Cosine-weighted gives denser sampling near the normal where the
			// BRDF response is strongest.
			Vec3 localView = SamplingDistributions::cosineSampleHemisphere(
				Sampler.next(), Sampler.next());
			Vec3 omega_o_world = viewFrame.toWorld(localView);

			// Safety: discard if it ended up below the normal (rounding)
			if (normal.dot(omega_o_world) <= 0.0f) continue;

			glm::vec3 omega_o = omega_o_world.ToGlm();

			collectSamplesForView(g, omega_o, 1.0f,
				Sampler, gaussians, bvh,
				BRDFSampleList_vec[threadID], threadID);
		}

		// Fix warm-start: reset to DC color after the view loop.
		// collectSamplesForView sets g.testAlbedo = SH(last_wo) on every call,
		// leaving it at a random Lo value instead of the intended base_color.
		// g.color = viewIndependent(g) = ZeroSH * SH_C0 + 0.5 = base_color.
		g.testAlbedo = glm::clamp(g.color.ToGlm(), 0.02f, 0.98f);
	}
}


// -------------------------------------------------------------------------
// collectAllSplatSamples_MT
//
// Divides the gaussians array evenly across threadNum threads.
// Replaces the old tile-based BRDF_MT for sample collection.
// -------------------------------------------------------------------------
void collectAllSplatSamples_MT(std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(threadNum);

	int total = (int)gaussians.size();
	int chunkSize = (total + threadNum - 1) / threadNum;

	std::vector<std::thread> threads;
	for (int t = 0; t < threadNum; ++t) {
		int start = t * chunkSize;
		int end = std::min(start + chunkSize, total);
		if (start >= total) break;
		threads.emplace_back(collectSplatSamples_Thread,
			t, start, end,
			std::ref(gaussians), bvh);
	}
	for (auto& th : threads)
		th.join();

	// Merge per-thread lists into the global list
	std::cout << "Combining BRDF samples from all threads...\n";
	for (int t = 0; t < threadNum; ++t)
		BRDFSampleList.insert(BRDFSampleList.end(),
			BRDFSampleList_vec[t].begin(), BRDFSampleList_vec[t].end());
}

void collectAllSplatSamples_ST(std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(1);

	int total = (int)gaussians.size();

	collectSplatSamples_Thread(0, 638299, 638300, std::ref(gaussians), bvh);

	// Merge per-thread lists into the global list
	std::cout << "Combining BRDF samples from all threads...\n";
	for (int t = 0; t < 1; ++t)
		BRDFSampleList.insert(BRDFSampleList.end(),
			BRDFSampleList_vec[t].begin(), BRDFSampleList_vec[t].end());
}

void collectSplatSamplesSubset_MT(std::vector<Gaussian>& gaussians, const std::vector<int>& indices, BVHNode* bvh)
{
	BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(threadNum);

	int total = (int)indices.size();
	int chunkSize = (total + threadNum - 1) / threadNum;

	std::vector<std::thread> threads;

	for (int t = 0; t < threadNum; ++t) {
		int start = t * chunkSize;
		int end = std::min(start + chunkSize, total);
		if (start >= total) break;

		threads.emplace_back(collectSplatSamples_Thread_Indexed,
			t, start, end,
			std::ref(gaussians),
			std::ref(indices),
			bvh);
	}

	for (auto& th : threads)
		th.join();

	std::cout << "Combining BRDF samples from all threads...\n";
	for (int t = 0; t < threadNum; ++t)
		BRDFSampleList.insert(BRDFSampleList.end(),
			BRDFSampleList_vec[t].begin(),
			BRDFSampleList_vec[t].end());
}


void renderBRDF(Camera& camera, std::vector<Gaussian>& gaussians, BVHNode* bvh, bool log = false)
{
	MTRandom Sampler;
	for (unsigned int y = 0; y < camera.height; y++) {
		for (unsigned int x = 0; x < camera.width; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, 0);
			BRDF(ray, bvh->getIntersectedGaussiansVec(0), gaussians, Sampler, bvh);
		}
	}
}

void renderBRDF_MT(int tileX, int tileY, int sizeX, int sizeY,
	int threadID, Camera& camera,
	std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	MTRandom Sampler;
	int startX = tileX * tileSize;
	int startY = tileY * tileSize;
	int endX = std::min(startX + tileSize, sizeX);
	int endY = std::min(startY + tileSize, sizeY);

	for (unsigned int y = startY; y < endY; y++) {
		for (unsigned int x = startX; x < endX; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, threadID);
			BRDF(ray, bvh->getIntersectedGaussiansVec(threadID),
				gaussians, Sampler, bvh, threadID);
		}
	}
}

void singleTest(Camera& camera, std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	MTRandom Sampler;
	float px = 25 + 0.5f, py = 25 + 0.5f;
	Ray ray = camera.generateRay(px, py);
	std::cout << "Ray stats: origin: " << ray.o.x << "," << ray.o.y << "," << ray.o.z
		<< " dir: " << ray.dir.x << "," << ray.dir.y << "," << ray.dir.z << "\n";
	BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(2);
	bvh->traverse(ray, gaussians, 1);
	BRDF(ray, bvh->getIntersectedGaussiansVec(1), gaussians, Sampler, bvh, 1);
}

// Kept for rendering tiles (image output only, no sample collection)
void BRDF_MT_Render(Camera& camera, std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	int tilesY = (camera.width + tileSize - 1) / tileSize;
	int tilesX = (camera.height + tileSize - 1) / tileSize;
	int totalTiles = tilesX * tilesY;

	std::atomic<int> tileNum(0);
	std::vector<std::thread> threads;

	auto threadFunc = [&](int threadID) {
		unsigned long i;
		while ((i = tileNum.fetch_add(1)) < totalTiles) {
			int tileX = i / tilesX;
			int tileY = i % tilesX;
			renderBRDF_MT(tileX, tileY, camera.width, camera.height,
				threadID, camera, gaussians, bvh);
		}
		};

	for (int i = 0; i < threadNum; i++)
		threads.emplace_back(threadFunc, i);
	for (auto& thread : threads)
		thread.join();
}

void setCamera(Camera& camera, RTCamera& viewCamera) {
	Vec3 from(0.0f, 0.0f, -5.0f);
	Vec3 to(0.0f, 0.0f, 0.0f);
	Vec3 up(0.0f, -1.0f, 0.0f);

	std::string data = "train";

	if (data == "train") {
		from = Vec3(-3.0f, -1.0f, -4.0f); 
		to = Vec3(-2.0f, 0.0f, 0.0f);
	}
	
	if (data == "tree") {
		from = Vec3(9.5f, 5.2f, -1.3f);
		to = Vec3(0.8f, 2.4f, 0.6f);
	}
	//Vec3 up(0.0f, 1.0f, 0.0f);
	viewCamera.from = from;
	viewCamera.to = to;
	viewCamera.up = up;
	viewCamera.setCamera(&camera);
}

void renderImageAlbedo(Camera& camera, GamesEngineeringBase::Window* canvas,
	std::vector<Gaussian>& gaussians, BVHNode* bvh) {
	int width = static_cast<int>(camera.width);
	int height = static_cast<int>(camera.height);

	for (unsigned int y = 0; y < height; y++) {
		for (unsigned int x = 0; x < width; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, 0);
			Colour color = GaussianAlbedo(ray, bvh->getIntersectedGaussiansVec(0));
			canvas->draw(x, y, color.r * 255.0f, color.g * 255.0f, color.b * 255.0f);
		}
		if (y % (height / 10) == 0)
			std::cout << "Rendering progress: " << (y * 100 / height) << "%\r";
	}
}

void renderImageSH(Camera& camera, GamesEngineeringBase::Window* canvas,
	std::vector<Gaussian>& gaussians, BVHNode* bvh) {
	int width = static_cast<int>(camera.width);
	int height = static_cast<int>(camera.height);
	int contribution_count = 0;

	for (unsigned int y = 0; y < height; y++) {//height
		for (unsigned int x = 0; x < width; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, 0);
			Colour color = GaussianColor(ray, bvh->getIntersectedGaussiansVec(0), contribution_count);
			canvas->draw(x, y, color.r * 255.0f, color.g * 255.0f, color.b * 255.0f);
		}
		if (y % (height / 10) == 0)
			std::cout << "Rendering progress: " << (y * 100 / height) << "%\r";
	}
}

void renderImageNormal(Camera& camera, GamesEngineeringBase::Window* canvas,
	std::vector<Gaussian>& gaussians, BVHNode* bvh) {
	int width = static_cast<int>(camera.width);
	int height = static_cast<int>(camera.height);
	int contribution_count = 0;

	for (unsigned int y = 0; y < height; y++) {//height
		for (unsigned int x = 0; x < width; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, 0);
			//Colour color = GaussianNormal(ray, bvh->getIntersectedGaussiansVec(0), contribution_count);
			Colour color = GaussianNormal_Single(ray, bvh->getIntersectedGaussiansVec(0), contribution_count);
			canvas->draw(x, y, color.r * 255.0f, color.g * 255.0f, color.b * 255.0f);
		}
		if (y % (height / 10) == 0)
			std::cout << "Rendering progress: " << (y * 100 / height) << "%\r";
	}
}

void renderOnePount(Camera& camera, GamesEngineeringBase::Window* canvas,
	std::vector<Gaussian>& gaussians, BVHNode* bvh) {//just to see contribuuting gaussians to a ray
	int width = static_cast<int>(camera.width);
	int height = static_cast<int>(camera.height);
	int contribution_count = 0;

	for (unsigned int y = 150; y < 151; y++) {//height
		for (unsigned int x = 300; x < 301; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, 0);
			Colour color = GaussianColor(ray, bvh->getIntersectedGaussiansVec(0), contribution_count);
			canvas->draw(x, y, color.r * 255.0f, color.g * 255.0f, color.b * 255.0f);
		}
		if (y % (height / 10) == 0)
			std::cout << "Rendering progress: " << (y * 100 / height) << "%\r";
	}
}

void getImportantGaussians(Camera& camera, GamesEngineeringBase::Window* canvas,
	std::vector<Gaussian>& gaussians, BVHNode* bvh, std::vector<int> &important_Gaussians) {
	int width = static_cast<int>(camera.width);
	int height = static_cast<int>(camera.height);
	int contribution_count = 0;
	std::set<int> important_gaussians_set;

	for (unsigned int y = 0; y < height; y++) {
		for (unsigned int x = 0; x < width; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, 0);

			std::vector<Gaussian>& intersected_gaussians = bvh->getIntersectedGaussiansVec(0);
			for(Gaussian g : intersected_gaussians) {
				important_gaussians_set.insert(g.index);
			}
		}
	}
	important_Gaussians = std::vector<int>(important_gaussians_set.begin(), important_gaussians_set.end());
}

void readAlbedoCSV(std::string filename, std::vector<Gaussian>& all) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Could not open the file - '" << filename << "'" << std::endl;
		return;
	}
	std::string line;
	std::getline(file, line); // skip header
	while (std::getline(file, line)) {
		std::istringstream ss(line);
		std::string value;
		std::vector<float> values;
		while (std::getline(ss, value, ','))
			values.push_back(std::stof(value));
		int index = static_cast<int>(values[0]);
		if (values.size() >= 4)
			all[index].testAlbedo = glm::vec3(values[1], values[2], values[3]);
	}
}

void getAlbedoCSV(std::vector<glm::vec3> albedos, std::vector<Gaussian>& all) {
	for (size_t i = 0; i < albedos.size(); i++)
		all[i].testAlbedo = albedos[i];
}

void renderBRDF2(Camera& camera, std::vector<Gaussian>& gaussians, BVHNode* bvh, bool log = false)
{
	MTRandom Sampler;
	int middleY = 34, middleX = 52;
	Colour color;

	for (unsigned int y = middleY; y < middleY + 1; y++) {
		for (unsigned int x = middleX; x < middleX + 1; x++) {
			float px = x + 0.5f, py = y + 0.5f;
			Ray ray = camera.generateRay(px, py);
			bvh->traverse(ray, gaussians, 0);
			color = BRDF(ray, bvh->getIntersectedGaussiansVec(0), gaussians, Sampler, bvh);
			std::cout << "Rendering pixel (" << x << ", " << y << ") - Color: ("
				<< color.r * 255.0f << ", " << color.g * 255.0f << ", " << color.b * 255.0f << ")\n";
		}
	}
	std::cout << "Final Color at center pixel: (" << color.r << ", " << color.g << ", " << color.b << ")\n";
}



// -------------------------------------------------------------------------
// spherePointTest
//
// Minimal, physics-correct single-point BRDF recovery test.
//
// Scene: unit sphere at origin + solid-colour background. No Gaussians.
// The hit point P is obtained by ray-tracing a camera ray into the sphere,
// giving an analytic normal N = normalize(P - sphere_centre).
// Hemisphere samples from P query the background (or return 0 for self-hits),
// producing clean Lo / Li pairs that the optimizer can fit.
//
// HOW TO RUN:
//   1.  python SphereSampleGen.py          → generates sphere_samples.csv
//   2.  (compile and) run this executable  → optimizer reads the CSV and
//       prints recovered base_color, metallic, roughness, specular
//
// Expected recovery (from SphereSampleGen.py ground truth):
//   base_color = (0.8, 0.2, 0.1)
//   metallic   = 0.0
//   roughness  = 0.3
//   specular   = 0.5
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// spherePointTestPLY
//
// PLY-pipeline version of the sphere point test.
// Loads sphere_scene.ply (generated by SpherePLYGen.py), which contains
// sphere-surface Gaussians + a sun Gaussian.  collectAllSplatSamples_ST
// already targets only splat 0 (front-face sphere point) via the
// `if (gi != 0) continue;` guard in collectSplatSamples_Thread.
//
// HOW TO RUN:
//   1.  python SpherePLYGen.py       → generates sphere_scene.ply
//   2.  (compile and) run this       → recovers BRDF from splat 0
// -------------------------------------------------------------------------
void spherePointTestPLY()
{
	std::cout << "=== Sphere Point BRDF Recovery Test (PLY pipeline) ===\n\n";
	std::cout << "  Step 1: Loading sphere_scene.ply\n";
	std::cout << "          (generate with: python SpherePLYGen.py)\n\n";

	std::vector<Gaussian> gaussians;
	//parsePLY("sphere_scene.ply", gaussians, "sphere_scene.ply");
	parsePLY("TreeHill.ply", gaussians, "TreeHillNormal.ply");
	if (gaussians.empty()) {
		std::cerr << "  ERROR: No Gaussians loaded. Run SpherePLYGen.py first.\n";
		return;
	}
	std::cout << "  Loaded " << gaussians.size() << " Gaussians (splat 0 = front sphere point)\n\n";

	std::cout << "  Step 2: Building BVH...\n";
	BVHNode bvh;
	bvh.build(gaussians);

	std::cout << "  Step 3: Collecting hemisphere samples for splat 0...\n";
	collectAllSplatSamples_ST(gaussians, &bvh);
	std::cout << "  Collected " << BRDFSampleList.size() << " samples\n\n";

	std::cout << "  Write Samples to file\n";
	writeBRDFSamples("BRDF_Samples.csv", gaussians);


	std::cout << "  Step 4: Optimising Disney BRDF...\n";
	optimizeDisneyBRDFAutodiff(BRDFSampleList, gaussians, /*maxIter=*/1000);
}


// -------------------------------------------------------------------------
// spherePointTestCSV
//
// CSV-mode version: reads Lo/Li directly from sphere_samples.csv (generated
// by SphereSampleGenFromPLY.py) and loads the PLY only to get the DC warm-
// start.  Loss(GT) = 0 exactly, so bc/metallic/roughness/specular all
// converge to ground-truth values given enough iterations.
//
// Run order:
//   1. python SpherePLYGen.py          → sphere_scene.ply
//   2. python SphereSampleGenFromPLY.py → sphere_samples.csv  (Loss(GT)=0)
//   3. Compile & run this executable   → bc converges to GT
// -------------------------------------------------------------------------
void spherePointTestCSV()
{
	std::cout << "=== Sphere Point BRDF Recovery Test (CSV mode — Loss(GT)=0) ===\n\n";

	// Load PLY to extract DC warm-start color for splat 0
	std::vector<Gaussian> gaussians;
	parsePLY("sphere_scene.ply", gaussians, "sphere_scene.ply");
	glm::vec3 warmStart(0.5f);  // fallback if PLY not loaded
	if (!gaussians.empty()) {
		// DC color = ZeroSH * SH_C0 + 0.5 = base_color
		warmStart = glm::clamp(
			glm::vec3(gaussians[0].ZeroSH.x * SH_C0 + 0.5f,
			          gaussians[0].ZeroSH.y * SH_C0 + 0.5f,
			          gaussians[0].ZeroSH.z * SH_C0 + 0.5f),
			0.02f, 0.98f);
		std::cout << "  Warm-start bc from PLY DC: ("
		          << warmStart.r << ", " << warmStart.g << ", " << warmStart.b << ")\n\n";
	}

	// Load sphere_samples.csv (Lo from same-samples GT, Li traced through PLY)
	std::cout << "  Loading sphere_samples.csv ...\n";
	std::vector<BRDFSample> samples = loadSphereSamples("sphere_samples.csv");

	if (samples.empty()) {
		std::cerr << "\n  ERROR: No samples loaded.\n"
		          << "  Please run SphereSampleGenFromPLY.py first.\n";
		return;
	}

	// Override warm-start in optimizeSpherePoint (which uses 0.5 grey)
	// by directly calling optimizeDisneyBRDFAutodiff with the gaussians
	// we loaded from the PLY (testAlbedo set to DC color above).
	for (auto& g : gaussians)
		g.testAlbedo = warmStart;

	std::cout << "\n  Optimising Disney BRDF (CSV mode, Loss(GT)=0)...\n";
	// Pass samples as BRDFSampleList and gaussians as context for warm-start
	optimizeDisneyBRDFAutodiff(samples, gaussians, /*maxIter=*/3000);
}


// -------------------------------------------------------------------------
// spherePointTest (CSV version — uses SpherePointOptim.h, legacy)
// -------------------------------------------------------------------------
void spherePointTest()
{
	std::cout << "=== Sphere Point BRDF Recovery Test ===\n\n";
	std::cout << "  Step 1: Loading samples from sphere_samples.csv\n";
	std::cout << "          (generate with: python SphereSampleGen.py)\n\n";

	std::vector<BRDFSample> samples = loadSphereSamples("sphere_samplesFROMPY.csv");

	if (samples.empty()) {
		std::cerr << "\n  ERROR: No samples loaded.\n"
		          << "  Please run SphereSampleGen.py first.\n";
		return;
	}

	std::cout << "\n  Step 2: Optimising Disney BRDF for single sphere point …\n";
	optimizeSpherePoint(
		samples,
		/*maxIter=*/ 3000,
		/*lr=*/      0.01f,
		/*verbose=*/ true);
}


int main() {

	// CSV mode (Loss(GT)=0 guaranteed — bc/rough/spec all converge to GT):
	//   1. python SpherePLYGen.py          → sphere_scene.ply
	//   2. python SphereSampleGenFromPLY.py → sphere_samples.csv
	//   3. Recompile and run
	//spherePointTestCSV();

	// BVH mode (Lo from SH — bc converges to ~0.76 due to L3 SH spectral limit):
	//   Uncomment below and comment out spherePointTestCSV() to use BVH mode.
	//spherePointTestPLY();

	//return 0;

	std::cout << "Parsing PLY file...\n";
	std::vector<Gaussian> gaussians{};
	//parsePLY("test_scene_rough.ply", gaussians, "test_scene_rough.ply");
	parsePLY("train.ply", gaussians, "trainWNormal.ply");
	std::cout << "Done PLY file...\n";

	float width = 500, height = 500, fov = 45;
	Matrix P = Matrix::perspective(0.001f, 10000.0f, (float)width / (float)height, fov);

	RTCamera viewCamera;
	Camera camera;
	camera.init(P, width, height);
	setCamera(camera, viewCamera);

	std::cout << "Building BVH...\n";
	BVHNode bvh;
	bvh.build(gaussians);
	std::cout << "Done building BVH...\n";


	std::cout << "Rendering final image using Spherical Harmonics...\n";
	GamesEngineeringBase::Window canvas;
	canvas.create((int)width, (int)height, "BRDF Optimization");
	
	//renderOnePount(camera, &canvas, gaussians, &bvh);
	//return 0;
	
	//renderImageSH(camera, &canvas, gaussians, &bvh);

	//rays.clear();
	//getRays(0, 42685, 42686, std::ref(gaussians), &bvh);
	//drawRays(rays, camera, &canvas, 0.5f, 255, 0, 255);

	renderImageNormal(camera, &canvas, gaussians, &bvh);

	savePNG("Train_normal_single.png", &canvas);
	return 0; 

	//std::vector<int> important_gaussians;
	//getImportantGaussians(camera, nullptr, gaussians, &bvh, important_gaussians);

	std::cout << "Obtaining BRDF samples (splat-loop, no camera rays)...\n";
	collectAllSplatSamples_ST(gaussians, &bvh);
	std::cout << "Done obtaining BRDF samples. Total: " << BRDFSampleList.size() << " samples\n";

	std::cout << "Writing BRDF samples to CSV...\n";
	writeBRDFSamples("BRDF_Samples.csv", gaussians);
	std::cout << "Done writing BRDF samples to CSV.\n";

	std::cout << "Optimizing BRDF parameters...\n";
	//optimizeDisneyBRDFAutodiff(BRDFSampleList, gaussians, 10000);
	//optimizeDisneyBRDFAutodiff(BRDFSampleList, gaussians, 10000);
	std::cout << "Done optimizing.\n";

	//writeBRDFSamples("BRDF_Correctfull.csv", gaussians);

	std::cout << "Rendering final image using Spherical Harmonics...\n";
	GamesEngineeringBase::Window canvas2;
	canvas2.create((int)width, (int)height, "BRDF Optimization");
	renderImageSH(camera, &canvas2, gaussians, &bvh);
	savePNG("Johnson.png", &canvas2);
	std::cout << "Done.\n";

	return 0;
}
