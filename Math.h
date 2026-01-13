#pragma once

// Do not change this code!

#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>
#include <memory.h>
#include <vector>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>


// Stop warnings about M_PI being a double
#pragma warning( disable : 4244)

#define SQ(x) (x * x)

#define threadNum 12

#define EPSILON 0.001f
#define M_EPSILON 0.00001f

float sigmoid(float x) {
	return 1.0 / (1.0 + std::exp(-x));
}

class Vec3
{
public:
	union {
		struct {
			float x;
			float y;
			float z;
			float w;
		};
		float coords[4];
	};
	Vec3()
	{
		x = 0;
		y = 0;
		z = 0;
		w = 1.0f;
	}
	Vec3(float _x)
	{
		x = _x;
		y = _x;
		z = _x;
		w = 1.0f;
	}
	Vec3(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
		w = 1.0f;
	}
	Vec3(float _x, float _y, float _z, float _w)
	{
		x = _x;
		y = _y;
		z = _z;
		w = _w;
	}
	Vec3 operator+(const Vec3 v) const
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}
	Vec3 operator-(const Vec3 v) const
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}
	Vec3 operator*(const float v) const
	{
		return Vec3(x * v, y * v, z * v);
	}
	Vec3 operator/(const float v) const
	{
		return Vec3(x / v, y / v, z / v, w / v);
	}
	Vec3 operator*(const Vec3 v) const
	{
		return Vec3(x * v.x, y * v.y, z * v.z);
	}
	Vec3 perspectiveDivide() const
	{
		return Vec3(x / w, y / w, z / w, 1.0f / w);
	}
	Vec3 operator-() const { return Vec3(-x, -y, -z); }
	float lengthSq()
	{
		return ((x * x) + (y * y) + (z * z));
	}
	float length()
	{
		return sqrtf((x * x) + (y * y) + (z * z));
	}
	Vec3 normalize() const
	{
		float l = 1.0f / sqrtf((x * x) + (y * y) + (z * z));
		return Vec3(x * l, y * l, z * l);
	}
	float dot(Vec3 v) const
	{
		return ((x * v.x) + (y * v.y) + (z * v.z));
	}
	Vec3 cross(Vec3 v) const
	{
		return Vec3((y * v.z) - (z * v.y), (z * v.x) - (x * v.z), (x * v.y) - (y * v.x));
	}
	Vec3 exponent() const {
		return Vec3(expf(x), expf(y), expf(z));
	}

	Vec3 Sigmoid() const {
		return Vec3(sigmoid(x), sigmoid(y), sigmoid(z));
	}

	float _max()const {
		//return (x > y ? (x > z ? x : z) : (y > z ? y : z));
		return std::max(std::max(x, y), z);
	}

	float _min()const {
		//return (x < y ? (x < z ? x : z) : (y < z ? y : z));
		return std::min(std::min(x, y), z);
	}

	glm::vec3 ToGlm()
	{
		return glm::vec3(x, y, z);
	}
};

class Colour
{
public:
	float r;
	float g;
	float b;
	Colour() { r = 0; g = 0; b = 0; }
	Colour(float _r, float _g, float _b)
	{
		r = _r;
		g = _g;
		b = _b;
	}
	Colour(unsigned char _r, unsigned char _g, unsigned char _b, unsigned char _a)
	{
		r = (float)_r / 255.0f;
		g = (float)_g / 255.0f;
		b = (float)_b / 255.0f;
	}
	void ToRGB(unsigned char& cr, unsigned char& cg, unsigned char& cb)
	{
		cr = (unsigned char)(r * 255);
		cg = (unsigned char)(g * 255);
		cb = (unsigned char)(b * 255);
	}

	void correct() {
		if (std::isnan(r) || r < 0) r = 0;
		if (std::isnan(g) || g < 0) g = 0;
		if (std::isnan(b) || b < 0) b = 0;
		if (r > 1.f) r = 1.f;
		if (g > 1.f) g = 1.f;
		if (b > 1.f) b = 1.f;
	}
	Colour operator+(const Colour& colour) const
	{
		Colour c;
		c.r = r + colour.r;
		c.g = g + colour.g;
		c.b = b + colour.b;
		return c;
	}

	Colour& operator+=(const Vec3& v) { r += v.x; g += v.y; b += v.z; return *this; }

	Colour operator-(const Colour& colour) const
	{
		Colour c;
		c.r = r - colour.r;
		c.g = g - colour.g;
		c.b = b - colour.b;
		return c;
	}
	Colour operator*(const Colour& colour) const
	{
		Colour c;
		c.r = r * colour.r;
		c.g = g * colour.g;
		c.b = b * colour.b;
		return c;
	}
	Colour operator/(const Colour& colour) const
	{
		Colour c;
		c.r = r / colour.r;
		c.g = g / colour.g;
		c.b = b / colour.b;
		return c;
	}
	Colour operator*(const float v) const
	{
		Colour c;
		c.r = r * v;
		c.g = g * v;
		c.b = b * v;
		return c;
	}
	Colour operator/(const float v) const
	{
		Colour c;
		c.r = r / v;
		c.g = g / v;
		c.b = b / v;
		return c;
	}
	float Lum()
	{
		return ((0.2126f * r) + (0.7152f * g) + (0.0722f * b));
	}

	Colour normalize() const
	{
		float l = 1.0f / sqrtf((r * r) + (g * g) + (b * b));
		return Colour(r * l, g * l, b * l);

		//float maxValue = std::max(std::max(r, g), b);
		//return maxValue > 0.0f ? Colour(r / maxValue, g / maxValue, b / maxValue) : Colour(0.0f, 0.0f, 0.0f);

	}

	glm::vec3 ToGlm() const
	{
		return glm::vec3(r, g, b);
	}
};

static float Dot(const Vec3 v1, const Vec3 v2)
{
	return ((v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z));
}


static Vec3 Cross(const Vec3& v1, const Vec3& v2)
{
	return Vec3((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x));
}

static Vec3 Max(Vec3 a, Vec3 b)
{
	return Vec3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
}

static Vec3 Min(Vec3 a, Vec3 b)
{
	return Vec3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
}

static Vec3 FromVector(std::vector<float> vec) {
	return Vec3(vec[0], vec[1], vec[2]);
}

static Vec3 fromGLM(glm::vec3 in) {
	return Vec3(in.x, in.y, in.z);
}

static Colour fromGLMC(glm::vec3 in) {
	return Colour(in.x, in.y, in.z);
}


class Matrix
{
public:
	union
	{
		float a[4][4];
		float m[16];
	};
	Matrix() { identity(); }
	Matrix(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33)
	{
		a[0][0] = m00;
		a[0][1] = m01;
		a[0][2] = m02;
		a[0][3] = m03;
		a[1][0] = m10;
		a[1][1] = m11;
		a[1][2] = m12;
		a[1][3] = m13;
		a[2][0] = m20;
		a[2][1] = m21;
		a[2][2] = m22;
		a[2][3] = m23;
		a[3][0] = m30;
		a[3][1] = m31;
		a[3][2] = m32;
		a[3][3] = m33;
	}
	void identity()
	{
		memset(m, 0, 16 * sizeof(float));
		m[0] = 1.0f;
		m[5] = 1.0f;
		m[10] = 1.0f;
		m[15] = 1.0f;
	}
	Matrix transpose()
	{
		return Matrix(a[0][0], a[1][0], a[2][0], a[3][0],
			a[0][1], a[1][1], a[2][1], a[3][1],
			a[0][2], a[1][2], a[2][2], a[3][2],
			a[0][3], a[1][3], a[2][3], a[3][3]);
	}
	float& operator[](int index)
	{
		return m[index];
	}
	static Matrix translation(const Vec3& v)
	{
		Matrix mat;
		mat.a[0][3] = v.x;
		mat.a[1][3] = v.y;
		mat.a[2][3] = v.z;
		return mat;
	}
	static Matrix scaling(const Vec3& v)
	{
		Matrix mat;
		mat.m[0] = v.x;
		mat.m[5] = v.y;
		mat.m[10] = v.z;
		return mat;
	}
	Matrix mul(const Matrix& matrix) const
	{
		Matrix ret;

		ret.m[0] = m[0] * matrix.m[0] + m[1] * matrix.m[4] + m[2] * matrix.m[8] + m[3] * matrix.m[12];
		ret.m[1] = m[0] * matrix.m[1] + m[1] * matrix.m[5] + m[2] * matrix.m[9] + m[3] * matrix.m[13];
		ret.m[2] = m[0] * matrix.m[2] + m[1] * matrix.m[6] + m[2] * matrix.m[10] + m[3] * matrix.m[14];
		ret.m[3] = m[0] * matrix.m[3] + m[1] * matrix.m[7] + m[2] * matrix.m[11] + m[3] * matrix.m[15];
		ret.m[4] = m[4] * matrix.m[0] + m[5] * matrix.m[4] + m[6] * matrix.m[8] + m[7] * matrix.m[12];
		ret.m[5] = m[4] * matrix.m[1] + m[5] * matrix.m[5] + m[6] * matrix.m[9] + m[7] * matrix.m[13];
		ret.m[6] = m[4] * matrix.m[2] + m[5] * matrix.m[6] + m[6] * matrix.m[10] + m[7] * matrix.m[14];
		ret.m[7] = m[4] * matrix.m[3] + m[5] * matrix.m[7] + m[6] * matrix.m[11] + m[7] * matrix.m[15];
		ret.m[8] = m[8] * matrix.m[0] + m[9] * matrix.m[4] + m[10] * matrix.m[8] + m[11] * matrix.m[12];
		ret.m[9] = m[8] * matrix.m[1] + m[9] * matrix.m[5] + m[10] * matrix.m[9] + m[11] * matrix.m[13];
		ret.m[10] = m[8] * matrix.m[2] + m[9] * matrix.m[6] + m[10] * matrix.m[10] + m[11] * matrix.m[14];
		ret.m[11] = m[8] * matrix.m[3] + m[9] * matrix.m[7] + m[10] * matrix.m[11] + m[11] * matrix.m[15];
		ret.m[12] = m[12] * matrix.m[0] + m[13] * matrix.m[4] + m[14] * matrix.m[8] + m[15] * matrix.m[12];
		ret.m[13] = m[12] * matrix.m[1] + m[13] * matrix.m[5] + m[14] * matrix.m[9] + m[15] * matrix.m[13];
		ret.m[14] = m[12] * matrix.m[2] + m[13] * matrix.m[6] + m[14] * matrix.m[10] + m[15] * matrix.m[14];
		ret.m[15] = m[12] * matrix.m[3] + m[13] * matrix.m[7] + m[14] * matrix.m[11] + m[15] * matrix.m[15];

		return ret;
	}
	Matrix operator*(const Matrix& matrix)
	{
		return mul(matrix);
	}
	Vec3 mulVec(const Vec3& v)
	{
		return Vec3(
			(v.x * m[0] + v.y * m[1] + v.z * m[2]),
			(v.x * m[4] + v.y * m[5] + v.z * m[6]),
			(v.x * m[8] + v.y * m[9] + v.z * m[10]));
	}
	Vec3 mulPoint(const Vec3& v)
	{
		Vec3 v1 = Vec3(
			(v.x * m[0] + v.y * m[1] + v.z * m[2]) + m[3],
			(v.x * m[4] + v.y * m[5] + v.z * m[6]) + m[7],
			(v.x * m[8] + v.y * m[9] + v.z * m[10]) + m[11]);
		return v1;
	}
	Vec3 mulPointAndPerspectiveDivide(const Vec3& v)
	{
		Vec3 v1 = Vec3(
			(v.x * m[0] + v.y * m[1] + v.z * m[2]) + m[3],
			(v.x * m[4] + v.y * m[5] + v.z * m[6]) + m[7],
			(v.x * m[8] + v.y * m[9] + v.z * m[10]) + m[11]);
		float w;
		w = (m[12] * v.x) + (m[13] * v.y) + (m[14] * v.z) + m[15];
		w = 1.0f / w;
		return (v1 * w);
	}
	Matrix operator=(const Matrix& matrix)
	{
		memcpy(m, matrix.m, sizeof(float) * 16);
		return (*this);
	}
	Matrix invert() // Unrolled inverse from MESA library
	{
		Matrix inv;
		inv[0] = m[5] * m[10] * m[15] -
			m[5] * m[11] * m[14] -
			m[9] * m[6] * m[15] +
			m[9] * m[7] * m[14] +
			m[13] * m[6] * m[11] -
			m[13] * m[7] * m[10];
		inv[4] = -m[4] * m[10] * m[15] +
			m[4] * m[11] * m[14] +
			m[8] * m[6] * m[15] -
			m[8] * m[7] * m[14] -
			m[12] * m[6] * m[11] +
			m[12] * m[7] * m[10];
		inv[8] = m[4] * m[9] * m[15] -
			m[4] * m[11] * m[13] -
			m[8] * m[5] * m[15] +
			m[8] * m[7] * m[13] +
			m[12] * m[5] * m[11] -
			m[12] * m[7] * m[9];
		inv[12] = -m[4] * m[9] * m[14] +
			m[4] * m[10] * m[13] +
			m[8] * m[5] * m[14] -
			m[8] * m[6] * m[13] -
			m[12] * m[5] * m[10] +
			m[12] * m[6] * m[9];
		inv[1] = -m[1] * m[10] * m[15] +
			m[1] * m[11] * m[14] +
			m[9] * m[2] * m[15] -
			m[9] * m[3] * m[14] -
			m[13] * m[2] * m[11] +
			m[13] * m[3] * m[10];
		inv[5] = m[0] * m[10] * m[15] -
			m[0] * m[11] * m[14] -
			m[8] * m[2] * m[15] +
			m[8] * m[3] * m[14] +
			m[12] * m[2] * m[11] -
			m[12] * m[3] * m[10];
		inv[9] = -m[0] * m[9] * m[15] +
			m[0] * m[11] * m[13] +
			m[8] * m[1] * m[15] -
			m[8] * m[3] * m[13] -
			m[12] * m[1] * m[11] +
			m[12] * m[3] * m[9];
		inv[13] = m[0] * m[9] * m[14] -
			m[0] * m[10] * m[13] -
			m[8] * m[1] * m[14] +
			m[8] * m[2] * m[13] +
			m[12] * m[1] * m[10] -
			m[12] * m[2] * m[9];
		inv[2] = m[1] * m[6] * m[15] -
			m[1] * m[7] * m[14] -
			m[5] * m[2] * m[15] +
			m[5] * m[3] * m[14] +
			m[13] * m[2] * m[7] -
			m[13] * m[3] * m[6];
		inv[6] = -m[0] * m[6] * m[15] +
			m[0] * m[7] * m[14] +
			m[4] * m[2] * m[15] -
			m[4] * m[3] * m[14] -
			m[12] * m[2] * m[7] +
			m[12] * m[3] * m[6];
		inv[10] = m[0] * m[5] * m[15] -
			m[0] * m[7] * m[13] -
			m[4] * m[1] * m[15] +
			m[4] * m[3] * m[13] +
			m[12] * m[1] * m[7] -
			m[12] * m[3] * m[5];
		inv[14] = -m[0] * m[5] * m[14] +
			m[0] * m[6] * m[13] +
			m[4] * m[1] * m[14] -
			m[4] * m[2] * m[13] -
			m[12] * m[1] * m[6] +
			m[12] * m[2] * m[5];
		inv[3] = -m[1] * m[6] * m[11] +
			m[1] * m[7] * m[10] +
			m[5] * m[2] * m[11] -
			m[5] * m[3] * m[10] -
			m[9] * m[2] * m[7] +
			m[9] * m[3] * m[6];
		inv[7] = m[0] * m[6] * m[11] -
			m[0] * m[7] * m[10] -
			m[4] * m[2] * m[11] +
			m[4] * m[3] * m[10] +
			m[8] * m[2] * m[7] -
			m[8] * m[3] * m[6];
		inv[11] = -m[0] * m[5] * m[11] +
			m[0] * m[7] * m[9] +
			m[4] * m[1] * m[11] -
			m[4] * m[3] * m[9] -
			m[8] * m[1] * m[7] +
			m[8] * m[3] * m[5];
		inv[15] = m[0] * m[5] * m[10] -
			m[0] * m[6] * m[9] -
			m[4] * m[1] * m[10] +
			m[4] * m[2] * m[9] +
			m[8] * m[1] * m[6] -
			m[8] * m[2] * m[5];
		float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
		if (det == 0)
		{
			// This code should never be called. Add error handling if it is
			identity();
			det = 1.0f;
		}
		det = 1.0f / det;
		for (int i = 0; i < 16; i++)
		{
			inv[i] = inv[i] * det;
		}
		return inv;
	}
	static Matrix lookAt(const Vec3& from, const Vec3& to, const Vec3& up)
	{
		Matrix mat;
		Vec3 dir = (from - to).normalize();
		Vec3 left = up.cross(dir).normalize();
		Vec3 newUp = dir.cross(left);
		mat.a[0][0] = left.x;
		mat.a[0][1] = left.y;
		mat.a[0][2] = left.z;
		mat.a[1][0] = newUp.x;
		mat.a[1][1] = newUp.y;
		mat.a[1][2] = newUp.z;
		mat.a[2][0] = dir.x;
		mat.a[2][1] = dir.y;
		mat.a[2][2] = dir.z;
		mat.a[0][3] = -from.dot(left);
		mat.a[1][3] = -from.dot(newUp);
		mat.a[2][3] = -from.dot(dir);
		mat.a[3][3] = 1;
		return mat;
	}
	static Matrix perspective(const float n, const float f, float aspect, const float fov) // FOV in degrees, outputs transposed Matrix for DX
	{
		Matrix pers;
		memset(pers.m, 0, sizeof(float) * 16);
		float t = 1.0f / (tanf(fov * 0.5f * 3.141592654f / 180.0f));
		pers.a[0][0] = t / aspect;
		pers.a[1][1] = t;
		pers.a[2][2] = -f / (f - n);
		pers.a[2][3] = -(f * n) / (f - n);
		pers.a[3][2] = -1.0f;
		return pers;
	}
	static Matrix rotateX(float theta)
	{
		Matrix mat;
		float ct = cosf(theta);
		float st = sinf(theta);
		mat.m[5] = ct;
		mat.m[6] = st;
		mat.m[9] = -st;
		mat.m[10] = ct;
		return mat;
	}
	static Matrix rotateY(float theta)
	{
		Matrix mat;
		float ct = cosf(theta);
		float st = sinf(theta);
		mat.m[0] = ct;
		mat.m[2] = -st;
		mat.m[8] = st;
		mat.m[10] = ct;
		return mat;
	}
	static Matrix rotateZ(float theta)
	{
		Matrix mat;
		float ct = cosf(theta);
		float st = sinf(theta);
		mat.m[0] = ct;
		mat.m[1] = st;
		mat.m[4] = -st;
		mat.m[5] = ct;
		return mat;
	}
};

class Mat3
{
public:
	union
	{
		float a[3][3];
		float m[9];
	};
	Mat3() { identity(); }
	Mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
	{
		a[0][0] = m00;
		a[0][1] = m01;
		a[0][2] = m02;
		a[1][0] = m10;
		a[1][1] = m11;
		a[1][2] = m12;
		a[2][0] = m20;
		a[2][1] = m21;
		a[2][2] = m22;
	}
	void identity()
	{
		memset(m, 0, 9 * sizeof(float));
		m[0] = 1.0f;
		m[4] = 1.0f;
		m[8] = 1.0f;
	}

	void diagonal(Vec3 v)
	{
		memset(m, 0, 9 * sizeof(float));
		m[0] = v.x;
		m[4] = v.y;
		m[8] = v.z;
	}

	Mat3 transpose()
	{
		return Mat3(a[0][0], a[1][0], a[2][0],
			a[0][1], a[1][1], a[2][1],
			a[0][2], a[1][2], a[2][2]);
	}
	float& operator[](int index)
	{
		return m[index];
	}

	Mat3 mul(const Mat3& matrix) const
	{
		Mat3 ret;

		ret.m[0] = m[0] * matrix.m[0] + m[3] * matrix.m[1] + m[6] * matrix.m[2];
		ret.m[1] = m[1] * matrix.m[0] + m[4] * matrix.m[1] + m[7] * matrix.m[2];
		ret.m[2] = m[2] * matrix.m[0] + m[5] * matrix.m[1] + m[8] * matrix.m[2];

		ret.m[3] = m[0] * matrix.m[3] + m[3] * matrix.m[4] + m[6] * matrix.m[5];
		ret.m[4] = m[1] * matrix.m[3] + m[4] * matrix.m[4] + m[7] * matrix.m[5];
		ret.m[5] = m[2] * matrix.m[3] + m[5] * matrix.m[4] + m[8] * matrix.m[5];

		ret.m[6] = m[0] * matrix.m[6] + m[3] * matrix.m[7] + m[6] * matrix.m[8];
		ret.m[7] = m[1] * matrix.m[6] + m[4] * matrix.m[7] + m[7] * matrix.m[8];
		ret.m[8] = m[2] * matrix.m[6] + m[5] * matrix.m[7] + m[8] * matrix.m[8];

		return ret;
	}
	Mat3 operator*(const Mat3& matrix)
	{
		return mul(matrix);
	}
	Vec3 mulRowVec(const Vec3& v)
	{
		return Vec3(
			(v.x * m[0] + v.y * m[3] + v.z * m[6]),
			(v.x * m[1] + v.y * m[4] + v.z * m[7]),
			(v.x * m[2] + v.y * m[5] + v.z * m[8]));
	}

	Mat3 operator=(const Mat3& matrix)
	{
		memcpy(m, matrix.m, sizeof(float) * 9);
		return (*this);
	}

	static Mat3 QuartToMatrix(const Vec3& q) {
		float x = q.x, y = q.y, z = q.z, w = q.w;
		return Mat3{
			1 - 2 * y * y - 2 * z * z,  2 * x * y - 2 * z * w,      2 * x * z + 2 * y * w,
			2 * x * y + 2 * z * w,      1 - 2 * x * x - 2 * z * z,  2 * y * z - 2 * x * w,
			2 * x * z - 2 * y * w,      2 * y * z + 2 * x * w,      1 - 2 * x * x - 2 * y * y
		};
	}

	//invert
	Mat3 Inverse() const
	{
		Mat3 result;

		float det =
			a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
			a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
			a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

		if (fabs(det) < 1e-8f)
		{
			// Singular matrix, cannot invert
			std::cerr << "Warning: Matrix is not invertible.\n";
			return result;  // Return identity or zero matrix
		}

		float invDet = 1.0f / det;

		result.a[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * invDet;
		result.a[0][1] = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]) * invDet;
		result.a[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * invDet;

		result.a[1][0] = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]) * invDet;
		result.a[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * invDet;
		result.a[1][2] = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]) * invDet;

		result.a[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * invDet;
		result.a[2][1] = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]) * invDet;
		result.a[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * invDet;

		return result;
	}

};

class Ray
{
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray()
	{
	}
	Ray(Vec3 _o, Vec3 _d)
	{
		init(_o, _d);
	}
	void init(Vec3 _o, Vec3 _d)
	{
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}
	Vec3 at(const float t) const
	{
		return (o + (dir * t));
	}
};




class AABB
{
public:
	Vec3 max;
	Vec3 min;
	AABB()
	{
		reset();
	}
	void reset()
	{
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}

	void set(Vec3 a, Vec3 b)
	{
		max = Max(a, b);
		min = Min(a, b);
	}

	void extend(const Vec3 p)
	{
		max = Max(max, p);
		min = Min(min, p);
	}

	void extend(const AABB a)
	{
		extend(a.max);
		extend(a.min);
	}
	// Add code here
	bool rayAABB(const Ray& r, float& t)
	{
		Vec3 tmin = (min - r.o) * r.invDir;
		Vec3 tmax = (max - r.o) * r.invDir;
		Vec3 Tentry = Min(tmin, tmax);
		Vec3 Texit = Max(tmin, tmax);
		float tentry = (std::max)(Tentry.x, (std::max)(Tentry.y, Tentry.z));
		float texit = (std::min)(Texit.x, (std::min)(Texit.y, Texit.z));
		t = (std::min)(tentry, texit);
		return (tentry <= texit);
	}
	// Add code here
	bool rayAABB(const Ray& r)
	{
		Vec3 tmin = (min - r.o) * r.invDir;
		Vec3 tmax = (max - r.o) * r.invDir;
		Vec3 Tentry = Min(tmin, tmax);
		Vec3 Texit = Max(tmin, tmax);
		float tentry = (std::max)(Tentry.x, (std::max)(Tentry.y, Tentry.z));
		float texit = (std::min)(Texit.x, (std::min)(Texit.y, Texit.z));

		return (tentry <= texit);
	}
	// Add code here
	float area()
	{
		Vec3 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
	}
};

struct Gaussian {
	Vec3 pos;
	Vec3 ZeroSH;
	Vec3 rotation;//Quarternions
	Vec3 scale;
	float opacity;
	Colour color; // color of the gaussian, used for rendering
	std::vector<float> higherSH;
	AABB aabb; // when reading we sgoukd compute this from the position and scale


	glm::mat3 R;
	glm::mat3 S2;
	glm::mat3 SinvR;

	glm::vec3 normal;
	glm::vec3 GaussNormal;

	glm::vec3 testAlbedo;

	void compute_gaussian_aabb() {
		float max_sigma = std::expf(scale._max());
		Vec3 radius(3.0f * max_sigma);
		aabb.set(pos - radius, pos + radius);
	}

	void compute_gaussian_covariance() {
		//rotation = rotation.normalize();
		Mat3 rotationMatrix = Mat3::QuartToMatrix(rotation);

		Vec3 _scale = scale.exponent();
		//_scale = _scale * 100.f;
		//_scale = _scale.normalize();

		Mat3 scaleMatrix;
		scaleMatrix.diagonal(_scale);

		glm::quat _rotation = glm::quat(rotation.x, rotation.y, rotation.z, rotation.w);
		//glm::quat _rotation = glm::quat(rotation.w ,rotation.x, rotation.y, rotation.z);
		_rotation = glm::normalize(_rotation);
		R = glm::mat3_cast(_rotation);

		glm::mat3 S2 = glm::mat3(1.0f);
		S2[0][0] = 1 / _scale.x;
		S2[1][1] = 1 / _scale.y;
		S2[2][2] = 1 / _scale.z;

		SinvR = R * S2;

	}

	void estimateNormal(Ray& ray) {
		Vec3 _scale = scale.exponent();

		//find smallest scale axis
		int minAxis = 0;
		glm::vec3 _scales = _scale.ToGlm();
		if (_scales.y < _scales.x) {
			minAxis = 1;
		}
		if (_scales.z < _scales[minAxis]) {
			minAxis = 2;
		}

		normal = glm::vec3(0.f, 0.f, 0.f);
		normal[minAxis] = 1.f;

		normal = glm::normalize(R * normal);

		glm::vec3 viewDir = -(ray.dir.ToGlm());
		if (glm::dot(normal, viewDir) < 0.0f) {
			normal = -normal;
		}
	}

	float computeAlpha(const Ray& r)
	{
		glm::vec3 ray_o = glm::vec3(r.o.x, r.o.y, r.o.z);
		glm::vec3 ray_d = glm::vec3(r.dir.x, r.dir.y, r.dir.z);
		glm::vec3 mean3D = glm::vec3(pos.x, pos.y, pos.z);


		glm::vec3 o_g = SinvR * (ray_o - mean3D);
		glm::vec3 d_g = SinvR * ray_d;

		float d_dot = glm::dot(d_g, d_g);
		float t_max = -glm::dot(o_g, d_g) / glm::max(1e-6f, d_dot);

		glm::vec3 pos = ray_o + t_max * ray_d;
		glm::vec3 p_g = SinvR * (mean3D - pos); // better numerical stability

		float alpha = opacity * std::expf(-0.5f * glm::dot(p_g, p_g));
		return glm::min(0.99f, alpha);
	}

	//for debug
	int index;

	// Complementary error function approximation
	float erfc_approx(float x) {
		// Use std::erfc if available
		return std::erfc(x);
	}

};



struct IntersectionData
{
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

#define MAXNODE_TRIANGLES 8
#define TRAVERSE_COST 1.0f
#define TRIANGLE_COST 2.0f
#define BUILD_BINS 32


#define BINS_COUNT 8

const float C_bounds = 1.0f; // Cost of traversing a node
const float C_isect = 1.0f;  // Cost of a ray-triangle intersection


struct Bin {
	AABB bounds;
	int triCount = 0;
};

#include <unordered_map>
std::unordered_map<Gaussian*, int> triangleMap;
int maxDepth = 0;
std::vector<Gaussian> intersectedGaussians;
std::vector<std::vector<Gaussian>> intersectedGaussiansVec;

int maxNodeTri = 0;
int nodeAbove = 0;
int maxCheckSize = 8;

class BVHNode
{
public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;
	// This can store an offset and number of triangles in a global triangle list for example
	// But you can store this however you want!
	//unsigned int offset;
	int num;
	std::vector<Gaussian*> triangles;
	int maxD() {
		return maxDepth;
	}
	BVHNode()
	{
		r = NULL;
		l = NULL;
	}
	// Note there are several options for how to implement the build method. Update this as required
	void build(std::vector<Gaussian>& inputTriangles) {
		intersectedGaussiansVec = std::vector<std::vector<Gaussian>>(threadNum*3);
		std::vector<Gaussian*> TempTriangles;
		for (int i = 0; i < inputTriangles.size(); i++) {
			TempTriangles.push_back(&inputTriangles[i]);
			triangleMap[&inputTriangles[i]] = i;
		}
		RecursiveBuild(TempTriangles);
	}

	float findBestSplitPlane(int& axis, float& splitPos) {
		float bestCost = FLT_MAX;
		float parentArea = bounds.area();

		for (int a = 0; a < 3; a++) {
			float boundsMin = FLT_MAX;
			float boundsMax = -FLT_MAX;

			for (Gaussian* tri : triangles) {
				float centroidPos = tri->pos.coords[a];
				boundsMin = std::min(boundsMin, centroidPos);
				boundsMax = std::max(boundsMax, centroidPos);
			}

			if (boundsMin == boundsMax) continue;

			Bin bin[BINS_COUNT];
			float scale = BINS_COUNT / (boundsMax - boundsMin);

			for (Gaussian* tri : triangles) {
				float centroidPos = tri->pos.coords[a];
				int binIdx = (std::min)(BINS_COUNT - 1, static_cast<int>((centroidPos - boundsMin) * scale));
				bin[binIdx].triCount++;
				bin[binIdx].bounds.extend(tri->aabb);
			}

			// Gather data 7 planes between 8 bins
			float leftArea[BINS_COUNT - 1], rightArea[BINS_COUNT - 1];
			int leftCount[BINS_COUNT - 1], rightCount[BINS_COUNT - 1];
			AABB leftBox, rightBox;
			int leftSum = 0, rightSum = 0;

			for (int i = 0; i < BINS_COUNT - 1; i++) {
				leftSum += bin[i].triCount;
				leftCount[i] = leftSum;
				if (bin[i].triCount > 0) {
					leftBox.extend(bin[i].bounds);
					leftArea[i] = leftBox.area();
				}
				else leftArea[i] = 0;

				rightSum += bin[(BINS_COUNT - 1) - i].triCount;
				rightCount[(BINS_COUNT - 2) - i] = rightSum;
				if (bin[(BINS_COUNT - 1) - i].triCount > 0) {
					rightBox.extend(bin[(BINS_COUNT - 1) - i].bounds);
					rightArea[(BINS_COUNT - 2) - i] = rightBox.area();
				}
				else rightArea[(BINS_COUNT - 2) - i] = 0;
			}

			float BinScale = (boundsMax - boundsMin) / BINS_COUNT;
			for (int i = 0; i < BINS_COUNT - 1; i++) {
				// Calculate cost of splitting at this bin using SAH
				float cost = C_bounds + ((leftArea[i] / parentArea) * leftCount[i] * C_isect) + ((rightArea[i] / parentArea) * rightCount[i] * C_isect);
				if (cost < bestCost) {
					bestCost = cost;
					axis = a;
					splitPos = boundsMin + BinScale * (i + 1);
				}
			}
		}
		return bestCost;
	}

	float calculateNodeCost() {
		float area = bounds.area();
		return num * area;
	}

	//recursively build the bvh
	void RecursiveBuild(std::vector<Gaussian*>& inputTriangles)
	{
		// Add BVH building code here
		//change later to not have traibge pointer voetor
		num = inputTriangles.size();
		triangles.reserve(num);
		for (Gaussian* a : inputTriangles) {
			triangles.emplace_back(a);
		}

		if (num <= 1)return;

		for (Gaussian* a : triangles) {
			bounds.extend(a->aabb);
		}

		int axis = -1;
		float splitPos = 0.0f;
		float splitCost = findBestSplitPlane(axis, splitPos);
		float noSplitCost = calculateNodeCost();

		if (splitCost >= noSplitCost || axis == -1) //return;
		{
			//std::cout << "No split, using leaf node with " << num << " triangles.\n";
		}

		std::vector<Gaussian*> leftTriangles, rightTriangles;
		for (Gaussian* tri : triangles) {
			if (tri->pos.coords[axis] < splitPos) {
				leftTriangles.push_back(tri);
			}
			else {
				rightTriangles.push_back(tri);
			}
		}

		if (leftTriangles.empty() || rightTriangles.empty()) return;

		triangles.clear();

		l = new BVHNode();
		r = new BVHNode();

		// Recursively build children
		l->RecursiveBuild(leftTriangles);
		r->RecursiveBuild(rightTriangles);


	}

	std::vector<Gaussian>& getIntersectedGaussians() {
		return intersectedGaussians;
	}

	std::vector<Gaussian>& getIntersectedGaussiansVec(int threadID) {
		return intersectedGaussiansVec[threadID];
	}

	//traverses the bvn
	void traverse(const Ray& ray, const std::vector<Gaussian>& InputTriangles, IntersectionData& intersection, int threadID)
	{
		// Add BVH Traversal code here
		if (!bounds.rayAABB(ray)) return;

		if (l == nullptr && r == nullptr) {
			for (Gaussian* tri : triangles) {
				float t;
				if (tri->aabb.rayAABB(ray, t)) {
					//return false;
					intersectedGaussiansVec[threadID].push_back(*tri);
				}
			}
			return;
		}

		if (l) l->traverse(ray, InputTriangles, intersection, threadID);
		if (r) r->traverse(ray, InputTriangles, intersection, threadID);
	}

	void traverse(const Ray& ray, const std::vector<Gaussian>& InputTriangles, IntersectionData& intersection)
	{
		// Add BVH Traversal code here
		if (!bounds.rayAABB(ray)) return;

		if (l == nullptr && r == nullptr) {
			for (Gaussian* tri : triangles) {
				float t;
				if (tri->aabb.rayAABB(ray, t)) {
					//return false;
					intersectedGaussians.push_back(*tri);
				}
			}
			return;
		}

		if (l) l->traverse(ray, InputTriangles, intersection);
		if (r) r->traverse(ray, InputTriangles, intersection);
	}

	void printIntersectionList() {
		std::cout << "Intersected Gaussians:\n";
		for (Gaussian& g : intersectedGaussians) {
			std::cout << g.index << " ";
		}
		std::cout << "\n";
	}

	IntersectionData traverse(const Ray& ray, const std::vector<Gaussian>& triangles, int threadID)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		intersectedGaussiansVec[threadID].clear();
		traverse(ray, triangles, intersection, threadID);
		//printIntersectionList();
		return intersection;
	}

	IntersectionData traverse(const Ray& ray, const std::vector<Gaussian>& triangles)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		intersectedGaussians.clear();
		traverse(ray, triangles, intersection);
		//printIntersectionList();
		return intersection;
	}
	bool traverseVisible(const Ray& ray, const std::vector<Gaussian>& InputTriangles, const float maxT)
	{
		// Add visibility code here
		float t, u, v;
		if (!bounds.rayAABB(ray, t) || t > maxT) return true;

		if (l == nullptr && r == nullptr) {
			for (Gaussian* tri : triangles) {
				if (tri->aabb.rayAABB(ray, t)) {
					//return false;
				}
			}
			//return true;
		}

		if (l && !l->traverseVisible(ray, InputTriangles, maxT)) return false;
		if (r && !r->traverseVisible(ray, InputTriangles, maxT)) return false;
		return true;
	}

	//FOR DEBUGGING
	void print(int depth = 0) {
		// Indentation for visualizing hierarchy
		std::string indent(depth * 2, ' ');

		// Print node information
		std::cout << indent << (l == nullptr && r == nullptr ? "Leaf" : "Node") << " (num triangles: " << static_cast<int>(num) << ")\n";
		std::cout << indent << "Bounds: min=(" << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << "), "
			<< "max=(" << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << ")\n";

		// Print triangles if this is a leaf node
		if (l == nullptr && r == nullptr) {
			for (Gaussian* a : triangles) {
				std::cout << indent << "  Triangle " << triangleMap[a] << ": "
					<< "v0=(" << a->pos.x << ", " << a->pos.y << ", " << a->pos.z << "))\n";
			}
		}

		// Recursively print children
		if (l) {
			std::cout << indent << "Left child:\n";
			l->print(depth + 1);
		}
		if (r) {
			std::cout << indent << "Right child:\n";
			r->print(depth + 1);
		}
	}
	void print2(int depth = 0) {
		// Indentation for visualizing hierarchy
		maxDepth = (std::max)(maxDepth, depth);
		std::string indent(depth * 2, '-');

		// Print node information
		//std::cout << indent << (l == nullptr && r == nullptr ? "Leaf" : "Node") << " (num triangles: " << static_cast<int>(num) << ")\n";

		// Print triangles if this is a leaf node
		if (l == nullptr && r == nullptr) {
			for (Gaussian* a : triangles) {
				//std::cout << indent << "  Triangle " << triangleMap[a] << "\n ";
			}
		}

		// Recursively print children
		if (l) {
			//std::cout << indent << "Left child:\n";
			l->print2(depth + 1);
		}
		if (r) {
			//std::cout << indent << "Right child:\n";
			r->print2(depth + 1);
		}
	}


	void printStat() {
		std::cout << "maxdepth: " << maxD() << "\n";
		std::cout << "Max triangles in a node: " << maxNodeTri << "\n";
		std::cout << "Number of nodes with more than " << maxCheckSize << " triangles: " << nodeAbove << "\n";
	}

	void checkTraverse(int depth = 0)
	{
		maxDepth = (std::max)(maxDepth, depth);
		if (l == nullptr && r == nullptr) {
			if (triangles.size() > 1) {
				maxNodeTri = std::max(maxNodeTri, (int)triangles.size());
			}
			if (triangles.size() > maxCheckSize) {
				nodeAbove++;
			}
			return;
		}

		if (l) l->checkTraverse(depth + 1);
		if (r) r->checkTraverse(depth + 1);
	}
};

class Camera
{
public:
	Matrix projectionMatrix;
	Matrix inverseProjectionMatrix;
	Matrix camera;
	Matrix cameraToView;
	float width = 0;
	float height = 0;
	Vec3 origin;
	Vec3 viewDirection;
	float Afilm;
	void init(Matrix ProjectionMatrix, int screenwidth, int screenheight)
	{
		projectionMatrix = ProjectionMatrix;
		inverseProjectionMatrix = ProjectionMatrix.invert();
		width = (float)screenwidth;
		height = (float)screenheight;
		float Wlens = (2.0f / ProjectionMatrix.a[1][1]);
		float aspect = ProjectionMatrix.a[0][0] / ProjectionMatrix.a[1][1];
		float Hlens = Wlens * aspect;
		Afilm = Wlens * Hlens;
	}
	void updateView(Matrix V)
	{
		camera = V;
		cameraToView = V.invert();
		origin = camera.mulPoint(Vec3(0, 0, 0));
		viewDirection = inverseProjectionMatrix.mulPointAndPerspectiveDivide(Vec3(0, 0, 1));
		viewDirection = camera.mulVec(viewDirection);
		viewDirection = viewDirection.normalize();
	}
	// Add code here
	Ray generateRay(float x, float y)
	{
		Vec3 dir(0, 0, 1);

		float xc = ((2 * x) / (width - 1)) - 1;
		float yc = 1 - ((2 * y) / (height - 1));

		Vec3 pclip = Vec3(xc, yc, 0, 1);
		Vec3 pcamera = inverseProjectionMatrix.mulPointAndPerspectiveDivide(pclip);


		Vec3 dcamera = camera.mulPointAndPerspectiveDivide(pcamera);

		dir = dcamera - origin;


		return Ray(origin, dir);
	}
	bool projectOntoCamera(const Vec3& p, float& x, float& y)
	{
		Vec3 pview = cameraToView.mulPoint(p);
		Vec3 pproj = projectionMatrix.mulPointAndPerspectiveDivide(pview);
		x = (pproj.x + 1.0f) * 0.5f;
		y = (pproj.y + 1.0f) * 0.5f;
		if (x < 0 || x > 1.0f || y < 0 || y > 1.0f)
		{
			return false;
		}
		x = x * width;
		y = 1.0f - y;
		y = y * height;
		return true;
	}
};

class RTCamera
{
public:
	Vec3 from;
	Vec3 to;
	Vec3 up;
	Camera* camera = NULL;
	float movespeed = 1.0f;
	float rotspeed = 5.0f;
	RTCamera()
	{
		rotspeed = 5.0f;
	}
	void setCamera(Camera* cam)
	{
		camera = cam;
		updateCamera();
	}
	void forward()
	{
		Vec3 dir = to - from;
		dir = dir.normalize() * movespeed;
		from = from + dir;
		to = from + dir;
		updateCamera();
	}
	void back()
	{
		Vec3 dir = to - from;
		dir = dir.normalize() * movespeed;
		from = from - dir;
		to = from + dir;
		updateCamera();
	}
	void left()
	{
		Vec3 dir = to - from;
		dir = dir.normalize();

		float rad = rotspeed * (M_PI / 180.0f);
		float cosTheta = cosf(rad);
		float sinTheta = sinf(rad);
		Vec3 k = up;
		Vec3 rotated = (dir * cosTheta) + (k.cross(dir) * sinTheta) + (k * (k.dot(dir) * (1 - cosTheta)));
		dir.x = rotated.x;
		dir.y = rotated.y;
		dir.z = rotated.z;
		to = from + dir;
		updateCamera();
	}
	void right()
	{
		Vec3 dir = to - from;
		dir = dir.normalize();
		float rad = -rotspeed * (M_PI / 180.0f);
		float cosTheta = cosf(rad);
		float sinTheta = sinf(rad);
		Vec3 k = up;
		Vec3 rotated = (dir * cosTheta) + (k.cross(dir) * sinTheta) + (k * (k.dot(dir) * (1 - cosTheta)));
		dir.x = rotated.x;
		dir.y = rotated.y;
		dir.z = rotated.z;
		to = from + dir;
		updateCamera();
	}
	void flyUp()
	{
		Vec3 dir = up * movespeed;
		from = from + dir;
		to = to + dir;
		updateCamera();
	}
	void flyDown()
	{
		Vec3 dir = up * movespeed;
		from = from - dir;
		to = to - dir;
		updateCamera();
	}
	void updateCamera()
	{
		Matrix V = Matrix::lookAt(from, to, up);
		V = V.invert();
		camera->updateView(V);
	}
	void strafeLeft()
	{
		Vec3 dir = (to - from).normalize();  // forward direction
		Vec3 right = dir.cross(up).normalize(); // local right axis
		Vec3 offset = right * movespeed;

		from = from - offset;
		to = to - offset;
		updateCamera();
	}

	void strafeRight()
	{
		Vec3 dir = (to - from).normalize();
		Vec3 right = dir.cross(up).normalize();
		Vec3 offset = right * movespeed;

		from = from + offset;
		to = to + offset;
		updateCamera();
	}
};
