#include <iostream>
#include <fstream>
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
			<< gaussians[s.splatIndex].testAlbedo.x << ","
			<< gaussians[s.splatIndex].testAlbedo.y << ","
			<< gaussians[s.splatIndex].testAlbedo.z
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

Colour GaussianColor(Ray& ray, std::vector<Gaussian>& in, int& contribution_count, bool correct = true)
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
			color = color + (SHColor * alpha * tr);
			tr *= (1.0f - alpha);
			contribution_count++;
		}
	}
	if (correct) color.correct();
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
	const int N_SAMPLES = 12;

	Vec3 normal = fromGLM(g.GaussNormal);

	// Flip normal toward the camera (view direction), not toward world origin.
	// omega_o points FROM the splat TO the camera.
	if (normal.dot(fromGLM(omega_o)) < 0.0f)
		normal = normal * -1.f;

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
			contribution_count,
			false);  // false = no tonemapping -- we want raw radiance

		// Clamp to [0, 2]: kills negative SH ringing, keeps physical range
		glm::vec3 L_i = glm::clamp(LiColor.ToGlm(), glm::vec3(0.0f), glm::vec3(2.0f));

		// Store one sample per hemisphere direction.
		// omega_i is the individual sample direction (normalized).
		// L_i is the raw radiance in that direction.
		// NdotL is stored so the optimizer can use it in the rendering equation.
		// L_o is filled in by the caller (it's a property of the splat, not the sample).
		out.push_back({
			g.index,
			omega_i_world.ToGlm(),   // omega_i: normalized world-space direction
			omega_o,                 // omega_o: view direction (same for all samples of this splat/ray)
			g.GaussNormal,           // surface normal (stored as-is for the optimizer)
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
void collectSplatSamples_Thread(
	int threadID,
	int startIdx,
	int endIdx,
	std::vector<Gaussian>& gaussians,
	BVHNode* bvh)
{
	// View directions per splat. 16 gives good angular coverage for
	// roughness recovery; reduce to 8 if performance is a concern.
	const int N_VIEW_SAMPLES = 16;

	MTRandom Sampler(threadID + 1); // distinct seed per thread

	for (int gi = startIdx; gi < endIdx; ++gi) {
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
	Vec3 up(0.0f, 1.0f, 0.0f);
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

	for (unsigned int y = 0; y < height; y++) {
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


int main(int argc, const char* argv[]) {

	std::cout << "Parsing PLY file...\n";
	std::vector<Gaussian> gaussians{};
	parsePLY("test_scene_Diff.ply", gaussians, "test_scene_Diff.ply");
	std::cout << "Done PLY file...\n";

	float width = 10, height = 10, fov = 45;
	Matrix P = Matrix::perspective(0.001f, 10000.0f, (float)width / (float)height, fov);

	RTCamera viewCamera;
	Camera camera;
	camera.init(P, width, height);
	setCamera(camera, viewCamera);

	std::cout << "Building BVH...\n";
	BVHNode bvh;
	bvh.build(gaussians);
	std::cout << "Done building BVH...\n";

	std::cout << "Obtaining BRDF samples (splat-loop, no camera rays)...\n";
	collectAllSplatSamples_MT(gaussians, &bvh);
	std::cout << "Done obtaining BRDF samples. Total: " << BRDFSampleList.size() << " samples\n";

	std::cout << "Optimizing BRDF parameters...\n";
	optimizeDisneyBRDFAutodiff(BRDFSampleList, gaussians, 10000);
	std::cout << "Done optimizing.\n";

	writeBRDFSamples("BRDF_Correctfull.csv", gaussians);

	std::cout << "Rendering final image using Spherical Harmonics...\n";
	GamesEngineeringBase::Window canvas;
	canvas.create((int)width, (int)height, "BRDF Optimization");
	renderImageSH(camera, &canvas, gaussians, &bvh);
	savePNG("miniScene.png", &canvas);
	std::cout << "Done.\n";

	return 0;
}
