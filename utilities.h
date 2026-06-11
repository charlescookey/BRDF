#pragma once

#include "Math.h"
#include "Sampling.h"
#include "Imaging.h"

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


// Draw a small arrowhead at (ex,ey) pointing away from (sx,sy).
// headLen is in pixels; angle is the half-spread in radians.
static void drawArrowHead(GamesEngineeringBase::Window* canvas,
	int W, int H,
	int sx, int sy, int ex, int ey,
	unsigned char r, unsigned char g, unsigned char b,
	int headLen = 8, float angle = 0.5236f /*30 deg*/)
{
	float dx = (float)(sx - ex), dy = (float)(sy - ey); // points back along shaft
	float len = std::sqrt(dx * dx + dy * dy);
	if (len < 1e-4f) return;
	dx /= len; dy /= len;

	float cosA = std::cos(angle), sinA = std::sin(angle);
	// rotate ±angle
	auto wing = [&](float c, float s) {
		int wx = ex + (int)((dx * c - dy * s) * headLen);
		int wy = ey + (int)((dx * s + dy * c) * headLen);
		drawLine(canvas, W, H, ex, ey, wx, wy, r, g, b);
	};
	wing( cosA,  sinA);
	wing( cosA, -sinA);
}

// Draw a vector of Rays as line segments on the canvas.
// Each ray is drawn from its origin to origin + dir * rayLen.
// Uses camera.projectOntoCamera() directly — no custom camera math needed.
void drawRays(const std::vector<Ray>& rays,
	Camera& camera,
	GamesEngineeringBase::Window* canvas,
	float rayLen = 1.0f, bool rayHead = false,
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
		int isx = clampI(sx, 0, W - 1), isy = clampI(sy, 0, H - 1);
		int iex = clampI(ex, 0, W - 1), iey = clampI(ey, 0, H - 1);
		drawLine(canvas, W, H, isx, isy, iex, iey, r, g, b);
		if (rayHead)
			drawArrowHead(canvas, W, H, isx, isy, iex, iey, r, g, b);
	}
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