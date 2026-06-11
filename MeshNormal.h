#pragma once

#include "BrdfSampleUtility.h"
#include "mesh.h"

#include "BRDFSample.h"
#include "BRDF_Optim_AutoDiff.h"

//global defines


void monteCarloSamplingHit(
	MTRandom& Sampler,
	MeshHit& hit,
	std::vector<Gaussian>& all,
	BVHNode* bvh,
	const glm::vec3& omega_o,        // view direction, used for normal flip
	std::vector<BRDFSample>& out,    // one entry appended per valid sample
	float contribution,              // alpha*tr weight, passed through to sample
	 std::vector<Ray>& raysForHit, int threadID = 0)
{
	const int N_SAMPLES = 20;

	Vec3 normal = hit.normal;

	// Flip normal toward the camera (view direction), not toward world origin.
	// omega_o points FROM the splat TO the camera.
	if (normal.dot(fromGLM(omega_o)) < 0.0f)
		normal = normal * -1.f;

	glm::vec3 normal_glm = normal.ToGlm();
	Frame frame;
	frame.fromVector(normal);
	int cc = 1;
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
		newRay.init(hit.hitPoint + (omega_i_world * EPSILON), omega_i_world);
		bvh->traverse(newRay, all, threadID + threadNum * 2);
		if(cc == 1)
		raysForHit.push_back(newRay);
		cc++;
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
			0,
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



static void collectSamplesForView(
	MeshHit& hit,
	const glm::vec3& omega_o,   // view direction (splat -> camera), normalized
	float contribution,
	MTRandom& Sampler,
	std::vector<Gaussian>& all,
	BVHNode* bvh,
	std::vector<BRDFSample>& out,
	int threadID, std::vector<Ray>& raysForHit)
{
	Vec3   viewVec = fromGLM(omega_o);


	Ray newRay;
	//ray that goes towards the hitpoint from the sampled direction
	Vec3 rayOrigin = hit.hitPoint + (viewVec * EPSILON);
	newRay.init(rayOrigin, -viewVec);
	bvh->traverse(newRay, all, threadID);
	//raysForHit.push_back(newRay);

	//std::cout << "ray origin: " << rayOrigin.x << " , " << rayOrigin.y << " , " << rayOrigin.z <<"\t the hit point: " << hit.hitPoint.x << " , " << hit.hitPoint.y << " , " << hit.hitPoint.z << "\n";
	//  std::cout << "ray direction: " << newRay.dir.x << " , " << newRay.dir.y << " , " << newRay.dir.z <<"normal: " << hit.normal.x << " , " << hit.normal.y << " , " << hit.normal.z << "\n";

	int contribution_count = 0;
	Colour SHColor = GaussianColor(newRay,bvh->getIntersectedGaussiansVec(threadID),contribution_count); 

	glm::vec3 sh_display = glm::clamp(SHColor.ToGlm(), glm::vec3(0.0f), glm::vec3(10.0f));
	glm::vec3 L_o = sh_display;  // for debug visualization only -- not used in optimizer

	// --- hemisphere samples for L_i ---
	int samplesBefore = (int)out.size();
	monteCarloSamplingHit(Sampler, hit, all, bvh, omega_o, out, contribution, raysForHit, threadID);

	// Fill in the shared-per-view fields for every sample just added
	for (int i = samplesBefore; i < (int)out.size(); i++) {
		out[i].L_o = L_o;
		out[i].shColour = sh_display;
	}
}

void collectSplatSamples_Hit(int threadID, MeshHit& hit, std::vector<Gaussian>& gaussians, BVHNode* bvh, std::vector<Ray>& raysForHit)
{
	BRDFSampleList_vec = std::vector<std::vector<BRDFSample>>(1);
	const int N_VIEW_SAMPLES = 16;

	MTRandom Sampler(threadID + 1); // distinct seed per thread

	Vec3 normal = hit.normal;
	normal = normal.normalize();

	Frame viewFrame;
	viewFrame.fromVector(normal);

	for (int v = 0; v < N_VIEW_SAMPLES; ++v) {
		Vec3 localView = SamplingDistributions::cosineSampleHemisphere(
			Sampler.next(), Sampler.next());
		Vec3 omega_o_world = viewFrame.toWorld(localView);

		// Safety: discard if it ended up below the normal (rounding)
		if (normal.dot(omega_o_world) <= 0.0f) continue;

		glm::vec3 omega_o = omega_o_world.ToGlm();

		collectSamplesForView(hit, omega_o, 1.0f,
			Sampler, gaussians, bvh,
			BRDFSampleList_vec[threadID], threadID,raysForHit);
	}

	std::cout << "Combining BRDF samples from all threads...\n";
	for (int t = 0; t < 1; ++t)
		BRDFSampleList.insert(BRDFSampleList.end(),
			BRDFSampleList_vec[t].begin(), BRDFSampleList_vec[t].end());
}


int meshNormalMain() {
	std::cout << "Parsing OBJ file...\n";
	std::vector<MeshTriangle> tris;
	loadOBJ("meshAlbedo.obj", tris);

	std::cout << "Building Mesh BVH...\n";
	MeshBVH meshBVH;
	meshBVH.build(tris);
	std::cout << "Done building Mesh BVH...\n";




	float width = 500, height = 500, fov = 45;
	Matrix P = Matrix::perspective(0.001f, 10000.0f, (float)width / (float)height, fov);

	RTCamera viewCamera;
	Camera camera;
	camera.init(P, width, height);
	__setCamera(camera, viewCamera);

	float px = 250 + 0.5f, py = 250 + 0.5f;
	Ray ray = camera.generateRay(px, py);

	bool ifHit = false;
	MeshHit hit = MeshNormalHit(ray, meshBVH, ifHit);

	if (!ifHit) {
		std::cout << "Ray missed the mesh!.\n";
		return 0;
	}

	tris.clear();
	meshBVH.clear();

	std::cout << "Parsing PLY file...\n";
	std::vector<Gaussian> gaussians{};
	parsePLY("train_optimized_scene.ply", gaussians, "trainWNormal.ply");
	std::cout << "Done PLY file...\n";

	std::cout << "Building BVH...\n";
	BVHNode bvh;
	bvh.build(gaussians);
	std::cout << "Done building BVH...\n";



	std::cout << "Rendering final image using Spherical Harmonics...\n";
	GamesEngineeringBase::Window canvas;
	canvas.create((int)width, (int)height, "BRDF Optimization");

	std::vector<Ray> raysForHit;
	collectSplatSamples_Hit(0, hit, gaussians, &bvh, raysForHit);

	writeBRDFSamplesRays("BRDFSamplesRay.csv");

	renderImageSH(camera, &canvas, gaussians, &bvh);
	savePNG("trainEmpty.png", &canvas);

	drawRays(raysForHit, camera, &canvas, 1.0f, true);
	savePNG("trainRays.png", &canvas);
	
	optimizeDisneyBRDFAutodiff(BRDFSampleList, gaussians, 10000);

	return 0;
}