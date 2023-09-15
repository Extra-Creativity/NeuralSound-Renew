#include "common.h"
#include "gpu_memory.h"
#include "integrand.h"
#include <cstdio>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace nwob;
class Triangle
{
public:
    float3 v0, v1, v2;
    int3 indices;
    HOST_DEVICE Triangle(float3 *vertices, int3 indices)
        : indices(indices)
    {
        v0 = vertices[indices.x];
        v1 = vertices[indices.y];
        v2 = vertices[indices.z];
    }
    HOST_DEVICE inline float area() const
    {
        return 0.5f * length(cross(v1 - v0, v2 - v0));
    }
};

void face2face(const torch::Tensor &vertices_,
               const torch::Tensor &triangles_,
               torch::Tensor &matrix_,
               const float wave_number, bool is_double_layer)
{
    PotentialType type = is_double_layer ? DOUBLE_LAYER : SINGLE_LAYER;
    auto triangle_size = triangles_.size(0);
    auto vertice_size = vertices_.size(0);
    const int max_singular_num = 64;
    GPUMemory<int> singular_indices(triangle_size * max_singular_num);
    GPUMemory<int> singular_num(triangle_size);
    singular_num.memset(0);
    parallel_for(
        triangle_size * triangle_size,
        [triangle_size, max_singular_num, vertices = (float3 *)vertices_.data_ptr(),
         triangles = (int3 *)triangles_.data_ptr(), wave_number, type,
         matrix = matrix_.data_ptr<float>(), singular_num = singular_num.device_ptr(),
         singular_indices = singular_indices.device_ptr()] __device__(int idx)
        {
            int i = idx / triangle_size;
            int j = idx % triangle_size;
            Triangle src(vertices, triangles[i]);
            Triangle trg(vertices, triangles[j]);
            if (triangle_common_vertex_num(src.indices, trg.indices) == 0)
                matrix[idx] = face2FaceIntegrand(
                                  src, trg, wave_number, type)
                                  .real();
            else
            {
                int singular_idx = atomicAdd(singular_num + i, 1);
                singular_indices[i * max_singular_num + singular_idx] = j;
            }
        });

    parallel_for(
        triangle_size * max_singular_num,
        [triangle_size, max_singular_num, vertices = (float3 *)vertices_.data_ptr(),
         triangles = (int3 *)triangles_.data_ptr(), wave_number, type,
         matrix = matrix_.data_ptr<float>(), singular_num = singular_num.device_ptr(),
         singular_indices = singular_indices.device_ptr()] __device__(int idx)
        {
            int i = idx / max_singular_num;
            int j = singular_indices[idx];
            if (singular_num[i] <= idx % max_singular_num)
                return;
            Triangle src(vertices, triangles[i]);
            Triangle trg(vertices, triangles[j]);
            matrix[i * triangle_size + j] = face2FaceIntegrand(
                                                src, trg, wave_number, type)
                                                .real();
        });
}

void face2points(const torch::Tensor &vertices_,
                 const torch::Tensor &triangles_,
                 const torch::Tensor &points_,
                 torch::Tensor &matrix_,
                 const float wave_number, bool is_double_layer)
{
    PotentialType type = is_double_layer ? DOUBLE_LAYER : SINGLE_LAYER;
    auto triangle_size = triangles_.size(0);
    auto vertice_size = vertices_.size(0);
    auto point_size = points_.size(0);
    parallel_for(
        triangle_size * point_size,
        [triangle_size, point_size, vertices = (float3 *)vertices_.data_ptr(),
         triangles = (int3 *)triangles_.data_ptr(), points = (float3 *)points_.data_ptr(),
         wave_number, type, matrix = matrix_.data_ptr<float>()] __device__(int idx)
        {
            int i = idx / point_size;
            int j = idx % point_size;
            Triangle src(vertices, triangles[i]);
            float3 trg = points[j];
            matrix[idx] = face2PointIntegrand(
                              src, trg, wave_number, type)
                              .real();
        });
}

void identity(const torch::Tensor &vertices_,
              const torch::Tensor &triangles_,
              torch::Tensor &matrix_)
{
    auto triangle_size = triangles_.size(0);
    auto vertice_size = vertices_.size(0);
    parallel_for(
        triangle_size,
        [triangle_size, vertices = (float3 *)vertices_.data_ptr(),
         triangles = (int3 *)triangles_.data_ptr(), area = matrix_.data_ptr<float>()] __device__(int idx)
        {
            int i = idx;
            Triangle src(vertices, triangles[i]);
            area[i] = src.area();
        });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("face2face", &face2face, "assemble face2face");
    m.def("face2points", &face2points, "assemble face2points");
    m.def("identity", &identity, "assemble identity");
}