#pragma once
#include "Math.h"
#include <random>
const float RAdius = 1.f;

class Sampler
{
public:
	virtual float next() = 0;
};

//onr per thread, seed diuff for diff thread, one mark
class MTRandom : public Sampler
{
public:
	std::mt19937 generator;
	std::uniform_real_distribution<float> dist;
	MTRandom(unsigned int seed = 1) : dist(0.0f, 1.0f)
	{
		generator.seed(seed);
	}
	float next()
	{
		return dist(generator);
	}
};


class Frame
{
public:
	Vec3 u;
	Vec3 v;
	Vec3 w;
	void fromVector(const Vec3& n)
	{
		// Gram-Schmit
		w = n.normalize();
		if (fabsf(w.x) > fabsf(w.y))
		{
			float l = 1.0f / sqrtf(w.x * w.x + w.z * w.z);
			u = Vec3(w.z * l, 0.0f, -w.x * l);
		}
		else
		{
			float l = 1.0f / sqrtf(w.y * w.y + w.z * w.z);
			u = Vec3(0, w.z * l, -w.y * l);
		}
		v = Cross(w, u);
	}

	void fromVectorTangent(const Vec3& n, const Vec3& t)
	{
		w = n.normalize();
		u = t.normalize();
		v = Cross(w, u);
	}
	Vec3 toLocal(const Vec3& vec) const
	{
		return Vec3(Dot(vec, u), Dot(vec, v), Dot(vec, w));
	}
	Vec3 toWorld(const Vec3& vec) const
	{
		return ((u * vec.x) + (v * vec.y) + (w * vec.z));
	}
};

class SamplingDistributions
{
public:
	static Vec3 uniformSampleHemisphere(float r1, float r2)
	{
		// Add code here
		float theta = acos(r1);
		float phi = 2 * M_PI * r2;

		float x = RAdius * sin(theta) * cos(phi);
		float y = RAdius * sin(theta) * sin(phi);
		float z = RAdius * cos(theta);

		return Vec3(x, y, z);
	}
	static float uniformHemispherePDF(const Vec3 wi)
	{
		// Add code here
		return 1.0f / 2 * M_PI;
	}
	static Vec3 cosineSampleHemisphere(float r1, float r2)
	{
		// Add code here
		float theta = acos(sqrtf(r1));
		float phi = 2 * M_PI * r2;

		float x = RAdius * sin(theta) * cos(phi);
		float y = RAdius * sin(theta) * sin(phi);
		float z = RAdius * cos(theta);

		return Vec3(x, y, z);;
	}
	static float cosineHemispherePDF(const Vec3 wi)
	{
		// Add code here
		return wi.z / M_PI;
	}
	static Vec3 uniformSampleSphere(float r1, float r2)
	{
		// Add code here
		float theta = acos(1 - (2 * r1));
		float phi = 2 * M_PI * r2;

		float x = RAdius * sin(theta) * cos(phi);
		float y = RAdius * sin(theta) * sin(phi);
		float z = RAdius * cos(theta);

		return Vec3(x, y, z);
	}
	static float uniformSpherePDF(const Vec3& wi)
	{
		// Add code here
		return 1.0f / 4 * M_PI;
	}
};
