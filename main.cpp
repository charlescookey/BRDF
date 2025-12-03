#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <iostream>


#include <fstream>
#include <thread>

#include "happly.h"

#include "GamesEngineeringBase.h"

#include "Math.h"
#include "Sampling.h"





#define SH_C0 0.28209479177387814f
#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

#define tileSize 16



struct BRDFSample {
	int splatIndex;
	glm::vec3 omega_i;
	glm::vec3 omega_o;
	glm::vec3 normal;
	glm::vec3 L_i;
	glm::vec3 L_o;
	float cosTheta;
	glm::vec3 fr;
	float weight;
};

std::vector<BRDFSample> BRDFSampleList;


void writeBRDFSamples(const std::string& filename, Colour colour) {
	std::ofstream file(filename);
	if (!file.is_open()) return;

	file << "splatIndex,omega_i.x,omega_i.y,omega_i.z,omega_o.x,omega_o.y,omega_o.z,normal.x,normal.y,normal.z,L_i.x,L_i.y,L_i.z,L_o.x,L_o.y,L_o.z,cosTheta,fr.r,fr.g,fr.b,weight\n";

	for (const auto& s : BRDFSampleList) {
		file << s.splatIndex << ","
			<< s.omega_i.x << "," << s.omega_i.y << "," << s.omega_i.z << ","
			<< s.omega_o.x << "," << s.omega_o.y << "," << s.omega_o.z << ","
			<< s.normal.x << "," << s.normal.y << "," << s.normal.z << ","
			<< s.L_i.x << "," << s.L_i.y << "," << s.L_i.z << ","
			<< s.L_o.x << "," << s.L_o.y << "," << s.L_o.z << ","
			<< s.cosTheta << ","
			<< s.fr.x << "," << s.fr.y << "," << s.fr.z << ","
			<< s.weight << "\n";
	}

	file << "\n Final color, " << colour.r << "," << colour.g << "," << colour.b << "\n";

	file.close();
	std::cout << "Saved " << BRDFSampleList.size()
		<< " BRDF samples to " << filename << std::endl;
}



Colour evaluateSphericalHarmonics(const Vec3& viewDir, Gaussian& gaussian) {
	Vec3 dir = viewDir.normalize();
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	float xx = x * x;
	float yy = y * y;
	float zz = z * z;
	float xy = x * y;
	float xz = x * z;
	float yz = y * z;

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

	//std::cout<<"Gaussian color : (" << gaussian.color.r << ", " << gaussian.color.g << ", " << gaussian.color.b << ")" << std::endl;
	return c;
}

Colour viewIndependent(Gaussian& gaussian) {

	Vec3 color = gaussian.ZeroSH * SH_C0;
	Colour c;
	c += color;
	c += Vec3(0.5f);
	return c;
}




void parsePLY(std::string filename, std::vector<Gaussian>& gaussians , std::string Normalfilename ) {
	happly::PLYData plyIn(filename.c_str());
	std::vector<float> elementA_prop1 = plyIn.getElement("vertex").getProperty<float>("x");
	std::vector<float> elementA_prop2 = plyIn.getElement("vertex").getProperty<float>("y");
	std::vector<float> elementA_prop3 = plyIn.getElement("vertex").getProperty<float>("z");

	int size = elementA_prop1.size();

	gaussians = std::vector<Gaussian>(size);

	for (size_t i = 0; i < elementA_prop1.size(); i++) {
		//std::cout << "x: " << elementA_prop1[i] << ", y: " << elementA_prop2[i] << ", z: " << elementA_prop3[i] << std::endl;
		gaussians[i].pos = Vec3(elementA_prop1[i], elementA_prop2[i], elementA_prop3[i]);
		gaussians[i].index = i;
	}

	happly::PLYData plyInNorm(Normalfilename.c_str());

	std::vector<double> elementA_prop1d = plyInNorm.getElement("vertex").getProperty<double>("nx");
	std::vector<double> elementA_prop2d = plyInNorm.getElement("vertex").getProperty<double>("ny");
	std::vector<double> elementA_prop3d = plyInNorm.getElement("vertex").getProperty<double>("nz");


	for (size_t i = 0; i < elementA_prop1.size(); i++) {
		gaussians[i].GaussNormal = glm::vec3(elementA_prop1d[i], elementA_prop2d[i], elementA_prop3d[i]);
	}


	elementA_prop1 = plyIn.getElement("vertex").getProperty<float>("f_dc_0");
	elementA_prop2 = plyIn.getElement("vertex").getProperty<float>("f_dc_1");
	elementA_prop3 = plyIn.getElement("vertex").getProperty<float>("f_dc_2");


	for (size_t i = 0; i < elementA_prop1.size(); i++) {
		gaussians[i].ZeroSH = Vec3(elementA_prop1[i], elementA_prop2[i], elementA_prop3[i]);
	}

	elementA_prop1 = plyIn.getElement("vertex").getProperty<float>("scale_0");
	elementA_prop2 = plyIn.getElement("vertex").getProperty<float>("scale_1");
	elementA_prop3 = plyIn.getElement("vertex").getProperty<float>("scale_2");

	for (size_t i = 0; i < elementA_prop1.size(); i++) {
		gaussians[i].scale = Vec3(elementA_prop1[i], elementA_prop2[i], elementA_prop3[i]);
		gaussians[i].compute_gaussian_aabb();
	}

	std::vector<float> elementA_prop4 = plyIn.getElement("vertex").getProperty<float>("opacity");

	for (size_t i = 0; i < elementA_prop1.size(); i++) {
		gaussians[i].opacity = sigmoid(elementA_prop4[i]);
	}

	elementA_prop1 = plyIn.getElement("vertex").getProperty<float>("rot_0");
	elementA_prop2 = plyIn.getElement("vertex").getProperty<float>("rot_1");
	elementA_prop3 = plyIn.getElement("vertex").getProperty<float>("rot_2");
	elementA_prop3 = plyIn.getElement("vertex").getProperty<float>("rot_3");

	for (size_t i = 0; i < elementA_prop1.size(); i++) {
		gaussians[i].rotation = Vec3(elementA_prop1[i], elementA_prop2[i], elementA_prop3[i], elementA_prop4[i]);
		gaussians[i].compute_gaussian_covariance();
	}


	for (int i = 0; i < 45; i++) {
		std::string name = "f_rest_" + std::to_string(i);

		elementA_prop1 = plyIn.getElement("vertex").getProperty<float>(name);

		for (size_t i = 0; i < elementA_prop1.size(); i++) {
			gaussians[i].higherSH.push_back(elementA_prop1[i]);
		}
	}
}

void setCamera(Camera& camera, RTCamera& viewCamera) {
	Vec3 from(-2.30947, 0.03062, -2.83864);
	viewCamera.from = from;

	Vec3 to = Vec3(-2.48386, 0.03176, -3.09662);
	viewCamera.to = to;

	Vec3 up(1.0f, 0.0f, 0.0f);
	viewCamera.up = up;

	viewCamera.setCamera(&camera);
}

void setCamera(Camera& camera, RTCamera& viewCamera, Vec3 from, Vec3 to, Vec3 up) {
	viewCamera.from = from;
	viewCamera.to = to;
	viewCamera.up = up;

	viewCamera.setCamera(&camera);
}

Colour GaussianColor(Ray& ray, std::vector<Gaussian>& in)
{
	struct Hit { float t; Gaussian* g; };
	std::vector<Hit> hits; hits.reserve(in.size());

	for (auto& g : in) {
		float t = ray.dir.dot((g.pos - ray.o));
		if (t <= 0) continue;               // for gaussian behind camera
		hits.push_back({ t, &g });
	}
	std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
		return a.t < b.t;                   // for closest to far
		});

	Colour color(0, 0, 0);
	float tr = 1.f;

	for (auto& h : hits) {
		Gaussian& g = *h.g;

		float alpha = g.computeAlpha(ray);

		if (tr < 0.001f) {
			break; // Stop if transmittance is very low
		}

		Vec3 viewDir = (ray.o - g.pos).normalize();
		Colour SHColor = evaluateSphericalHarmonics(viewDir, g);

		color = color + (SHColor * alpha * tr);
		tr *= (1.0f - alpha); // Update transmittance
	}
	color.correct();
	return color;
}


glm::vec3 monteCarloSampling(MTRandom& Sampler, Gaussian& g, std::vector<Gaussian>& all, BVHNode* bvh, float& cosTheta) {
	// --- Monte Carlo hemisphere sampling for incoming radiance ---
	const int N_SAMPLES = 16;
	glm::vec3 L_i_sum(0.0f);
	float hemisphere_weight_sum = 0.0f;

	for (int s = 0; s < N_SAMPLES; ++s) {
		Vec3 localDir = SamplingDistributions::cosineSampleHemisphere(Sampler.next(), Sampler.next());
		Frame frame;
		Vec3 normal = fromGLM(g.normal);
		frame.fromVector(normal);

		Vec3 omega_i = frame.toWorld(localDir);

		// Create secondary ray from splat center
		Ray newRay;
		newRay.init(g.pos + (omega_i * EPSILON), omega_i);


		// Evaluate radiance along this direction (reusing your existing path)
		// The function can use your BVH traversal or Gaussian intersection
		bvh->traverse(newRay, all, 1);
		Colour LiColor = GaussianColor(newRay, bvh->getIntersectedGaussiansVec(1));
		glm::vec3 L_i = LiColor.ToGlm();

		float cosTheta_i = glm::dot(g.normal, omega_i.ToGlm());
		L_i_sum += L_i * cosTheta_i;
		hemisphere_weight_sum += cosTheta_i;
	}

	// Monte Carlo estimate of hemisphere-integrated incoming radiance
	glm::vec3 L_i_MC = (hemisphere_weight_sum > 0.0f)
		? (L_i_sum * 3.14159265358979323846f / hemisphere_weight_sum)
		: glm::vec3(0.0f);
	cosTheta = hemisphere_weight_sum / N_SAMPLES;
	return L_i_MC;
}


Colour BRDF(Ray& ray, std::vector<Gaussian>& in, std::vector<Gaussian>& all, MTRandom& Sampler, BVHNode* bvh)
{
	struct Hit { float t; Gaussian* g; };
	std::vector<Hit> hits; hits.reserve(in.size());

	for (auto& g : in) {
		float t = ray.dir.dot((g.pos - ray.o));
		if (t <= 0) continue;               // for gaussian behind camera
		hits.push_back({ t, &g });
	}
	std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
		return a.t < b.t;                   // for closest to far
		});

	Colour color(0, 0, 0);
	float tr = 1.f;

	for (auto& h : hits) {
		Gaussian& g = *h.g;
		Colour accumBefore = color;

		float alpha = g.computeAlpha(ray);

		if (tr < 0.001f) {
			break; // Stop if transmittance is very low
		}

		Vec3 viewDir = (ray.o - g.pos).normalize();
		Colour SHColor = evaluateSphericalHarmonics(viewDir, g);
		glm::vec3 omega_o = viewDir.ToGlm();
		glm::vec3 L_o = SHColor.ToGlm();
		//g.estimateNormal(ray);
		//glm::vec3 normal = g.normal;
		glm::vec3 normal = g.GaussNormal;

		glm::vec3 omega_i = -(ray.dir.normalize().ToGlm());


		//geoetric term
		float NxOmega = glm::dot(normal, omega_i);

		glm::vec3 L_i = accumBefore.ToGlm();


		float contribution = alpha * tr;

		if (contribution > 0.05f) {
			glm::vec3 L_e(0, 0, 0); // assume zero unless you have emission data

			// per-channel fr estimate:
			glm::vec3 fr_est;
			for (int c = 0; c < 3; ++c) {
				float numerator = L_o[c] - L_e[c];
				float denom = L_i[c] * NxOmega;
				fr_est[c] = numerator / denom;
			}

			// clamp unrealistic values
			fr_est.r = glm::clamp(fr_est.r, 0.0f, 1000.0f);
			fr_est.g = glm::clamp(fr_est.g, 0.0f, 1000.0f);
			fr_est.b = glm::clamp(fr_est.b, 0.0f, 1000.0f);

			MTRandom Sampler;
			float cosTheta_i;
			glm::vec3 L_i = monteCarloSampling(Sampler, g, all, bvh, cosTheta_i);

			// store sample: splat index, omega_i, omega_o(viewDir), normal, fr_est, weight
			BRDFSampleList.push_back({ g.index ,omega_i , omega_o, normal , L_i , L_o,cosTheta_i, fr_est, contribution });
		}
		color = color + (SHColor * alpha * tr);
		tr *= (1.0f - alpha); // Update transmittance
	}
	color.correct();
	return color;
}

void renderBRDF(Camera& camera, std::vector<Gaussian>& gaussians, BVHNode* bvh)
{
	//for (unsigned int y = 0; y < camera.width; y++)
	//{
	//	for (unsigned int x = startX; x < endX; x++)
	//	{
	//		float px = x + 0.5f;
	//		float py = y + 0.5f;
	//		Ray ray = camera.generateRay(px, py);

	//		bvh->traverse(ray, gaussians, threadID);
	//		std::vector<PixelContribution> PC;
	//		Colour color = GaussianColor(ray, bvh->getIntersectedGaussiansVec(threadID), PC);

	//		int pxy = (y * sizeX) + x;
	//		pixelContributionsMap[pxy] = PC;


	//		//Colour color = GaussianColor(ray, gaussians);
	//		canvas->draw(x, y, color.r * 255.0f, color.g * 255.0f, color.b * 255.0f);
	//		film[(y * sizeX) + x] = color;
	//	}
	//}
	int x = camera.width / 2;
	int y = camera.height / 2;
	float px = x + 0.5f;
	float py = y + 0.5f;
	Ray ray = camera.generateRay(px, py);


	MTRandom Sampler;
	bvh->traverse(ray, gaussians, 0);
	std::vector<BRDFSample> Brdfs;
	Colour color = BRDF(ray, bvh->getIntersectedGaussiansVec(0), gaussians, Sampler, bvh);

	writeBRDFSamples("BRDF_full.csv", color);

}


int main(int argc, const char* argv[]) {
	std::cout << "Parsing PLY file...\n";
	std::vector<Gaussian> gaussians{};
	//parsePLY("point_cloud.ply", gaussians);
	parsePLY("train.ply", gaussians , "trainWNormal.ply");
	std::cout << "Done PLY file...\n";
	float width = 1024;
	float height = 1024;
	float fov = 45;



	Matrix P = Matrix::perspective(0.001f, 10000.0f, (float)width / (float)height, fov);

	Camera camera;
	camera.init(P, width, height);


	std::cout << "Building BVH...\n";
	BVHNode bvh;
	bvh.build(gaussians);
	std::cout << "Done building BVH...\n";


	renderBRDF(camera, gaussians, &bvh);

	return 0;
}

