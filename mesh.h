#pragma once

#include "Math.h"


#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cfloat>

#include "GamesEngineeringBase.h"
#include "Imaging.h"


// ============================================================
//  Vec2  (UV coordinates)
//  Mirrors the Vec3 style in Math.h
// ============================================================
class Vec2
{
public:
    float x, y;
    Vec2() : x(0), y(0) {}
    Vec2(float _x, float _y) : x(_x), y(_y) {}
    Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 operator*(float f)       const { return Vec2(x * f, y * f); }
};


// ============================================================
//  Image
//  Loads a PNG/JPG from disk (via stb_image) and lets you
//  sample it by UV coordinate, matching the Colour type used
//  everywhere else in the project.
// ============================================================
inline Colour sampleImage(const GamesEngineeringBase::Image& img, Vec2 uv)
{
    if (!img.data) return Colour(1, 0, 1);   // magenta = missing texture

    // Wrap UVs into [0, 1)
    float u = uv.x - floorf(uv.x);
    float v = uv.y - floorf(uv.y);

    // Flip V: OBJ UV origin is bottom-left, image is top-left
    v = 1.0f - v;

    int x = std::min((int)(u * img.width), (int)img.width - 1);
    int y = std::min((int)(v * img.height), (int)img.height - 1);

    int idx = (y * img.width + x) * img.channels;

    // WIC loads BGR(A) by default — channels 0=B, 1=G, 2=R
    return Colour(
        img.data[idx + 2] / 255.f,   // R
        img.data[idx + 1] / 255.f,   // G
        img.data[idx + 0] / 255.f    // B
    );
}

// ============================================================
//  MeshVertex / MeshTriangle
// ============================================================
struct MeshVertex
{
    Vec3 pos;
    Vec3 normal;
    Vec2 uv;
};

struct MeshTriangle
{
    MeshVertex v[3];

    Vec3 centroid() const {
        return (v[0].pos + v[1].pos + v[2].pos) * (1.f / 3.f);
    }

    // Axis-aligned bounding box helpers (used by MeshBVH)
    Vec3 aabbMin() const {
        return Vec3(
            (std::min)({ v[0].pos.x, v[1].pos.x, v[2].pos.x }),
            (std::min)({ v[0].pos.y, v[1].pos.y, v[2].pos.y }),
            (std::min)({ v[0].pos.z, v[1].pos.z, v[2].pos.z })
        );
    }
    Vec3 aabbMax() const {
        return Vec3(
            (std::max)({ v[0].pos.x, v[1].pos.x, v[2].pos.x }),
            (std::max)({ v[0].pos.y, v[1].pos.y, v[2].pos.y }),
            (std::max)({ v[0].pos.z, v[1].pos.z, v[2].pos.z })
        );
    }
};


// ============================================================
//  MeshHit  –  result of a ray-triangle test
// ============================================================
struct MeshHit
{
    bool  hit = false;
    float t = FLT_MAX;
    Vec3  normal;   // barycentric-interpolated, world space
    Vec2  uv;    // barycentric-interpolated texture coordinate
    Vec3 hitPoint;
};


// ============================================================
//  Möller–Trumbore intersection
//  Uses your existing Vec3 cross/dot, matches EPSILON from Math.h
// ============================================================
inline MeshHit intersectTriangle(const Ray& ray, const MeshTriangle& tri)
{
    MeshHit result;

    Vec3 e1 = tri.v[1].pos - tri.v[0].pos;
    Vec3 e2 = tri.v[2].pos - tri.v[0].pos;
    Vec3 h = ray.dir.cross(e2);
    float a = e1.dot(h);

    if (fabsf(a) < 1e-8f) return result;   // ray parallel to triangle

    float  f = 1.f / a;
    Vec3   s = ray.o - tri.v[0].pos;
    float  u = f * s.dot(h);
    if (u < 0.f || u > 1.f) return result;

    Vec3  q = s.cross(e1);
    float v = f * ray.dir.dot(q);
    if (v < 0.f || u + v > 1.f) return result;

    float t = f * e2.dot(q);
    if (t < EPSILON) return result;         // behind ray origin

    float w = 1.f - u - v;

    result.hit = true;
    result.t = t;
    result.normal = (tri.v[0].normal * w
        + tri.v[1].normal * u
        + tri.v[2].normal * v).normalize();
    result.uv = tri.v[0].uv * w
        + tri.v[1].uv * u
        + tri.v[2].uv * v;
	result.hitPoint = ray.at(t);
    return result;
}


// ============================================================
//  MeshBVH
//  Simple surface-area-heuristic BVH over MeshTriangles.
//  Mirrors the structure of BVHNode in Math.h so you can
//  use it the same way: build() once, then traverse() per ray.
// ============================================================

// Axis-aligned bounding box for triangles (reuses your AABB style)
struct MeshAABB
{
    Vec3 mn, mx;

    MeshAABB() : mn(FLT_MAX), mx(-FLT_MAX) {}
    MeshAABB(Vec3 a, Vec3 b) : mn(a), mx(b) {}

    void expand(const MeshAABB& o) {
        mn = Min(mn, o.mn);
        mx = Max(mx, o.mx);
    }

    // Slab test – same approach as rayAABB in Math.h
    bool rayIntersect(const Ray& ray, float& tOut) const
    {
        float tmin = 0.f, tmax = FLT_MAX;
        for (int i = 0; i < 3; i++) {
            float inv = 1.f / ray.dir.coords[i];
            float t0 = (mn.coords[i] - ray.o.coords[i]) * inv;
            float t1 = (mx.coords[i] - ray.o.coords[i]) * inv;
            if (inv < 0.f) std::swap(t0, t1);
            tmin = (std::max)(tmin, t0);
            tmax = (std::min)(tmax, t1);
            if (tmax < tmin) return false;
        }
        tOut = tmin;
        return true;
    }
};

class MeshBVH
{
public:
    MeshAABB              bounds;
    MeshBVH* left = nullptr;
    MeshBVH* right = nullptr;
    std::vector<MeshTriangle*> tris;   // only populated in leaf nodes

    ~MeshBVH() { delete left; delete right; }

    // Build over a flat list of triangles (pointers into your scene vector)
    void build(std::vector<MeshTriangle>& scene, int maxLeaf = 4)
    {
        std::vector<MeshTriangle*> ptrs(scene.size());
        for (size_t i = 0; i < scene.size(); i++) ptrs[i] = &scene[i];
        buildRecursive(ptrs, maxLeaf);
    }

    // Returns the nearest hit along the ray, or hit.hit == false if none.
    MeshHit traverse(const Ray& ray) const
    {
        float dummy;
        if (!bounds.rayIntersect(ray, dummy)) return MeshHit{};
        return traverseRecursive(ray);
    }
    void clear()
    {
        recursiveDelete();
        bounds = MeshAABB();
        tris.clear();
	}

    void recursiveDelete()
    {
        if (left) {
            left->recursiveDelete();
            delete left;
            left = nullptr;
        }
        if (right) {
            right->recursiveDelete();
            delete right;
            right = nullptr;
        }
	}

private:
    void buildRecursive(std::vector<MeshTriangle*>& in, int maxLeaf)
    {
        // Compute node bounds
        bounds = MeshAABB();
        for (auto* t : in) {
            MeshAABB tb(t->aabbMin(), t->aabbMax());
            bounds.expand(tb);
        }

        // Leaf condition
        if ((int)in.size() <= maxLeaf) {
            tris = in;
            return;
        }

        // Split on longest axis at centroid midpoint
        Vec3 extent = bounds.mx - bounds.mn;
        int  axis = 0;
        if (extent.y > extent.coords[axis]) axis = 1;
        if (extent.z > extent.coords[axis]) axis = 2;

        float mid = (bounds.mn.coords[axis] + bounds.mx.coords[axis]) * 0.5f;

        std::vector<MeshTriangle*> lTris, rTris;
        for (auto* t : in) {
            if (t->centroid().coords[axis] < mid)
                lTris.push_back(t);
            else
                rTris.push_back(t);
        }

        // Degenerate split guard
        if (lTris.empty() || rTris.empty()) {
            tris = in;
            return;
        }

        left = new MeshBVH(); left->buildRecursive(lTris, maxLeaf);
        right = new MeshBVH(); right->buildRecursive(rTris, maxLeaf);
    }

    MeshHit traverseRecursive(const Ray& ray) const
    {
        // Leaf: test all triangles, return nearest
        if (!left && !right) {
            MeshHit best;
            for (auto* t : tris) {
                MeshHit h = intersectTriangle(ray, *t);
                if (h.hit && h.t < best.t) best = h;
            }
            return best;
        }

        float tL = FLT_MAX, tR = FLT_MAX;
        bool  hitL = left && left->bounds.rayIntersect(ray, tL);
        bool  hitR = right && right->bounds.rayIntersect(ray, tR);

        if (!hitL && !hitR) return MeshHit{};

        // Visit nearer child first for early exit
        MeshBVH* __near = left, * __far = right;
        if (tR < tL) { std::swap(__near, __far); std::swap(hitL, hitR); }

        MeshHit h1 = __near ? __near->traverseRecursive(ray) : MeshHit{};
        // Skip far child if near hit is already closer than far AABB entry
        if (hitR && !(h1.hit && h1.t < tR)) {
            MeshHit h2 = __far->traverseRecursive(ray);
            if (h2.hit && h2.t < h1.t) return h2;
        }
        return h1;
    }
};

void validateBVH(const MeshBVH* node, int depth = 0, int maxLeaf = 4,
    int* maxDepth = nullptr, int* maxTrisInLeaf = nullptr,
    int* totalLeaves = nullptr)
{
    // Root call: create accumulators
    static int _maxDepth, _maxTris, _totalLeaves;
    if (depth == 0) {
        _maxDepth = 0; _maxTris = 0; _totalLeaves = 0;
        maxDepth = &_maxDepth;
        maxTrisInLeaf = &_maxTris;
        totalLeaves = &_totalLeaves;
    }

    if (!node) return;

    *maxDepth = std::max(*maxDepth, depth);

    if (!node->left && !node->right) {
        // Leaf node
        (*totalLeaves)++;
        int n = (int)node->tris.size();
        *maxTrisInLeaf = std::max(*maxTrisInLeaf, n);
        if (n > maxLeaf)
            std::cout << "  [WARN] Leaf at depth " << depth
            << " has " << n << " tris (over limit of " << maxLeaf << ")\n";
        return;
    }

    validateBVH(node->left, depth + 1, maxLeaf, maxDepth, maxTrisInLeaf, totalLeaves);
    validateBVH(node->right, depth + 1, maxLeaf, maxDepth, maxTrisInLeaf, totalLeaves);

    // Print summary at root return
    if (depth == 0) {
        std::cout << "[BVH] Max depth      : " << *maxDepth << "\n";
        std::cout << "[BVH] Total leaves   : " << *totalLeaves << "\n";
        std::cout << "[BVH] Max tris/leaf  : " << *maxTrisInLeaf << "\n";
    }
}

inline bool loadOBJ(const std::string& path, std::vector<MeshTriangle>& out)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[OBJ] Cannot open: " << path << "\n";
        return false;
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok; ss >> tok;

        if (tok == "v") {
            float x, y, z; ss >> x >> y >> z;
            positions.push_back(Vec3(x, y, z));
        }
        else if (tok == "vn") {
            float x, y, z; ss >> x >> y >> z;
            normals.push_back(Vec3(x, y, z));
        }
        else if (tok == "vt") {
            float u, v; ss >> u >> v;
            uvs.push_back(Vec2(u, v));
        }
        else if (tok == "f") {
            std::vector<MeshVertex> faceVerts;
            std::string ref;
            while (ss >> ref) {
                // Handles: v    v/vt    v//vn    v/vt/vn
                int vi = 0, vti = 0, vni = 0;
                if (ref.find("//") != std::string::npos)
                    sscanf(ref.c_str(), "%d//%d", &vi, &vni);
                else
                    sscanf(ref.c_str(), "%d/%d/%d", &vi, &vti, &vni);

                MeshVertex mv;
                if (vi > 0 && vi <= (int)positions.size()) mv.pos = positions[vi - 1];
                if (vni > 0 && vni <= (int)normals.size())   mv.normal = normals[vni - 1];
                if (vti > 0 && vti <= (int)uvs.size())       mv.uv = uvs[vti - 1];
                faceVerts.push_back(mv);
            }

            // Fan triangulation
            for (int i = 1; i + 1 < (int)faceVerts.size(); i++) {
                MeshTriangle tri;
                tri.v[0] = faceVerts[0];
                tri.v[1] = faceVerts[i];
                tri.v[2] = faceVerts[i + 1];

                // SuGaR exports no vn data — compute flat normal from geometry.
                // For dense meshes (200k+ tris) flat normals are visually fine.
                Vec3 e1 = tri.v[1].pos - tri.v[0].pos;
                Vec3 e2 = tri.v[2].pos - tri.v[0].pos;
                Vec3 n = e1.cross(e2).normalize();
                tri.v[0].normal = n;
                tri.v[1].normal = n;
                tri.v[2].normal = n;

                out.push_back(tri);
            }
        }
    }

    std::cout << "[OBJ] Loaded " << path << "  → " << out.size() << " triangles"
        << (normals.empty() ? "  (normals computed from geometry)\n" : "  (normals from file)\n");
    return true;
}

// ============================================================
//  MeshColor  –  drop-in replacement for GaussianColor()
//
//  Usage in main (mirrors your existing render calls):
//
//    Colour c = MeshColor(ray, meshBVH, albedoTex);
//
// ============================================================
inline Colour MeshColor(const Ray& ray, const MeshBVH& bvh, const GamesEngineeringBase::Image& albedo)
{
    MeshHit h = bvh.traverse(ray);
    if (!h.hit) return Colour(0, 0, 0);
    return sampleImage(albedo, h.uv);
}

// Normal-visualisation variant (matches your renderImageNormal pipeline)
inline Colour MeshNormal(const Ray& ray, const MeshBVH& bvh)
{
    MeshHit h = bvh.traverse(ray);
    if (!h.hit) return Colour(0, 0, 0);
    // Remap [-1,1] → [0,1] so it looks like your GaussianNormal output
    return Colour(
        h.normal.x * 0.5f + 0.5f,
        h.normal.y * 0.5f + 0.5f,
        h.normal.z * 0.5f + 0.5f
    );
}

inline MeshHit MeshNormalHit(const Ray& ray, const MeshBVH& bvh, bool &hit)
{
    MeshHit h = bvh.traverse(ray);
	hit = h.hit;
    if (!hit) return MeshHit();
    // Remap [-1,1] → [0,1] so it looks like your GaussianNormal output
    return h;
}

void renderMesh(Camera& camera, GamesEngineeringBase::Window* canvas,
    const MeshBVH& bvh, const GamesEngineeringBase::Image& albedo) {
    int width = static_cast<int>(camera.width);
    int height = static_cast<int>(camera.height);
    int contribution_count = 0;

    for (unsigned int y = 0; y < height; y++) {//height
        for (unsigned int x = 0; x < width; x++) {
            float px = x + 0.5f, py = y + 0.5f;
            Ray ray = camera.generateRay(px, py);

            Colour color = MeshNormal(ray,bvh);
            canvas->draw(x, y, color.r * 255.0f, color.g * 255.0f, color.b * 255.0f);
        }
        if (y % (height / 10) == 0)
            std::cout << "Rendering progress: " << (y * 100 / height) << "%\r";
    }
}

void __setCamera(Camera& camera, RTCamera& viewCamera) {
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


int main3() {

    std::cout << "Parsing OBJ file...\n";
    std::vector<MeshTriangle> tris;
    loadOBJ("meshAlbedo.obj", tris);

    GamesEngineeringBase::Image albedo;
    //albedo.load("meshAlbedo.png");

    std::cout << "Done PLY file...\n";

    float width = 500, height = 500, fov = 45;
    Matrix P = Matrix::perspective(0.001f, 10000.0f, (float)width / (float)height, fov);

    RTCamera viewCamera;
    Camera camera;
    camera.init(P, width, height);
    __setCamera(camera, viewCamera);

    std::cout << "Building BVH...\n";
    MeshBVH meshBVH;
    meshBVH.build(tris);
    validateBVH(&meshBVH);

    std::cout << "Done building BVH...\n";


    std::cout << "Rendering final image using Spherical Harmonics...\n";
    GamesEngineeringBase::Window canvas;
    canvas.create((int)width, (int)height, "BRDF Optimization");


    renderMesh(camera, &canvas, meshBVH, albedo);

    savePNG("Train_mesh_albedo.png", &canvas);
    return 0;
}