#pragma once
#include "line.h"
#include "triangle.h"
#include "potential.h"

NWOB_NAMESPACE_BEGIN
inline __device__ __host__ int triangle_common_vertex_num(int3 ind1, int3 ind2)
{
    return (ind1.x == ind2.x) + (ind1.x == ind2.y) + (ind1.x == ind2.z) + (ind1.y == ind2.x) + (ind1.y == ind2.y) +
           (ind1.y == ind2.z) + (ind1.z == ind2.x) + (ind1.z == ind2.y) + (ind1.z == ind2.z);
}

// normalized normal of a triangle
inline __device__ __host__ float3 triangle_norm(float3 *verts)
{
    float3 v1 = verts[1] - verts[0];
    float3 v2 = verts[2] - verts[0];
    float3 n = cross(v1, v2);
    return n / length(n);
}

inline __device__ __host__ float jacobian(float3 *v)
{
    return length(cross(v[1] - v[0], v[2] - v[0]));
}

inline __device__ __host__ float jacobian(float3 v1, float3 v2, float3 v3)
{
    return length(cross(v2 - v1, v3 - v1));
}

inline __device__ __host__ float jacobian(float3 *verts, int3 ind)
{
    return jacobian(verts[ind.x], verts[ind.y], verts[ind.z]);
}

// unit triangle (0, 0), (1, 0), (0, 1)
inline __device__ __host__ float3 local_to_global(float x1, float x2, float3 *v)
{
    return (1 - x1 - x2) * v[0] + x1 * v[1] + x2 * v[2];
}

// unit triangle (0, 0), (1, 0), (1, 1)
inline __device__ __host__ float3 local_to_global2(float x1, float x2, float3 *v)
{
    return (1 - x1) * v[0] + (x1 - x2) * v[1] + x2 * v[2];
}

inline __device__ __host__ complex singular_point(float xsi,
                                                  float eta1,
                                                  float eta2,
                                                  float eta3,
                                                  float weight,
                                                  float3 *trial_v,
                                                  float3 *test_v,
                                                  float3 trial_norm,
                                                  int neighbor_num,
                                                  complex s,
                                                  PotentialType type)
{
    complex result = complex(0, 0);
    xsi = 0.5 * (xsi + 1);
    eta1 = 0.5 * (eta1 + 1);
    eta2 = 0.5 * (eta2 + 1);
    eta3 = 0.5 * (eta3 + 1);
    switch (neighbor_num)
    {
    case 3:
    { // Indentical Panels
        float w = xsi * xsi * xsi * eta1 * eta1 * eta2;
        float eta12 = eta1 * eta2;
        float eta123 = eta1 * eta2 * eta3;
        float3 v1, v2;
        // Region 1
        v1 = local_to_global2(xsi, xsi * (1.0 - eta1 + eta12), trial_v);
        v2 = local_to_global2(xsi * (1.0 - eta123), xsi * (1.0 - eta1), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        // Region 2
        v1 = local_to_global2(xsi * (1.0 - eta123), xsi * (1.0 - eta1), trial_v);
        v2 = local_to_global2(xsi, xsi * (1.0 - eta1 + eta12), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        // Region 3
        v1 = local_to_global2(xsi, xsi * (eta1 - eta12 + eta123), trial_v);
        v2 = local_to_global2(xsi * (1.0 - eta12), xsi * (eta1 - eta12), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        // Region 4
        v1 = local_to_global2(xsi * (1.0 - eta12), xsi * (eta1 - eta12), trial_v);
        v2 = local_to_global2(xsi, xsi * (eta1 - eta12 + eta123), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        // Region 5
        v1 = local_to_global2(xsi * (1.0 - eta123), xsi * (eta1 - eta123), trial_v);
        v2 = local_to_global2(xsi, xsi * (eta1 - eta12), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        // Region 6
        v1 = local_to_global2(xsi, xsi * (eta1 - eta12), trial_v);
        v2 = local_to_global2(xsi * (1.0 - eta123), xsi * (eta1 - eta123), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        result = result * w * weight;
        break;
    }
    case 2:
    { // Common Edge
        float w = xsi * xsi * xsi * eta1 * eta1;
        float eta12 = eta1 * eta2;
        float eta123 = eta1 * eta2 * eta3;
        float3 v1, v2;
        // Region 1
        v1 = local_to_global2(xsi, xsi * eta1 * eta3, trial_v);
        v2 = local_to_global2(xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        // Region 2
        v1 = local_to_global2(xsi, xsi * eta1, trial_v);
        v2 = local_to_global2(xsi * (1.0 - eta123), xsi * eta1 * eta2 * (1 - eta3), test_v);
        result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
        // Region 3
        v1 = local_to_global2(xsi * (1.0 - eta12), xsi * eta1 * (1.0 - eta2), trial_v);
        v2 = local_to_global2(xsi, xsi * eta123, test_v);
        result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
        // Region 4
        v1 = local_to_global2(xsi * (1.0 - eta123), xsi * eta12 * (1.0 - eta3), trial_v);
        v2 = local_to_global2(xsi, xsi * eta1, test_v);
        result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
        // Region 5
        v1 = local_to_global2(xsi * (1.0 - eta123), xsi * eta1 * (1.0 - eta2 * eta3), trial_v);
        v2 = local_to_global2(xsi, xsi * eta12, test_v);
        result += layer_potential(v1, v2, trial_norm, s, type) * eta2;
        result = result * w * weight;
        break;
    }
    case 1:
    { // Common Vertex
        float w = xsi * xsi * xsi * eta2;
        float3 v1, v2;
        // Region 1
        v1 = local_to_global2(xsi, xsi * eta1, trial_v);
        v2 = local_to_global2(xsi * eta2, xsi * eta2 * eta3, test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        // Region 2
        v1 = local_to_global2(xsi * eta2, xsi * eta2 * eta3, trial_v);
        v2 = local_to_global2(xsi, xsi * eta1, test_v);
        result += layer_potential(v1, v2, trial_norm, s, type);
        result = result * w * weight;
        break;
    }
    }
    return result;
}

inline __device__ __host__ complex singular_integrand(float3 *trial_v,
                                                      float3 *test_v,
                                                      float trial_jacobian,
                                                      float test_jacobian,
                                                      float3 trial_norm,
                                                      complex s,
                                                      int neighbor_num,
                                                      PotentialType type)
{
    float guass_x[LINE_GAUSS_NUM] = LINE_GAUSS_XS;
    float guass_w[LINE_GAUSS_NUM] = LINE_GAUSS_WS;
    complex result = complex(0, 0);
    for (int xsi_i = 0; xsi_i < LINE_GAUSS_NUM; xsi_i++)
        for (int eta1_i = 0; eta1_i < LINE_GAUSS_NUM; eta1_i++)
            for (int eta2_i = 0; eta2_i < LINE_GAUSS_NUM; eta2_i++)
                for (int eta3_i = 0; eta3_i < LINE_GAUSS_NUM; eta3_i++)
                {
                    result += singular_point(guass_x[xsi_i], guass_x[eta1_i], guass_x[eta2_i], guass_x[eta3_i],
                                             guass_w[xsi_i] * guass_w[eta1_i] * guass_w[eta2_i] * guass_w[eta3_i],
                                             trial_v, test_v, trial_norm, neighbor_num, s, type);
                }
    return result * trial_jacobian * test_jacobian / 16;
}

inline __device__ __host__ complex regular_integrand(float3 *trial_v,
                                                     float3 *test_v,
                                                     float trial_jacobian,
                                                     float test_jacobian,
                                                     complex s,
                                                     PotentialType type)
{
    complex result = complex(0, 0);
    float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
    float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
    float3 trial_norm = triangle_norm(test_v);
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
        for (int j = 0; j < TRI_GAUSS_NUM; j++)
        {
            float3 v1 = local_to_global(guass_x[i][0], guass_x[i][1], trial_v);
            float3 v2 = local_to_global(guass_x[j][0], guass_x[j][1], test_v);
            result += 0.25 * guass_w[i] * guass_w[j] * trial_jacobian * test_jacobian *
                      layer_potential(v1, v2, trial_norm, s, type);
        }
    return result;
}

inline __device__ __host__ complex
potential_integrand(float3 point, float3 *src_v, float src_jacobian, complex s, PotentialType type)
{
    complex result = complex(0, 0);
    float guass_x[TRI_GAUSS_NUM][2] = TRI_GAUSS_XS;
    float guass_w[TRI_GAUSS_NUM] = TRI_GAUSS_WS;
    float3 src_norm = triangle_norm(src_v);
    for (int i = 0; i < TRI_GAUSS_NUM; i++)
    {
        float3 v_in_tri = local_to_global(guass_x[i][0], guass_x[i][1], src_v);
        result += 0.5 * guass_w[i] * src_jacobian * layer_potential(point, v_in_tri, src_norm, s, type);
    }
    return result;
}

template <class Tri>
inline __device__ __host__ complex face2FaceIntegrand(const Tri &src, const Tri &trg, complex k, PotentialType type)
{

    float3 src_v[3] = {{src.v0}, {src.v1}, {src.v2}};
    float src_jacobian = jacobian(src_v);
    float3 trg_v[3] = {{trg.v0}, {trg.v1}, {trg.v2}};
    float trg_jacobian = jacobian(trg_v);
    int neighbor_num = triangle_common_vertex_num(src.indices, trg.indices);
    if (neighbor_num == 0)
        return regular_integrand(src_v, trg_v, src_jacobian, trg_jacobian, k, type);
    else
    {
        float3 src_v_temp[3];
        float3 trg_v_temp[3];
        float3 src_norm = triangle_norm(trg_v);
        for (int i = 0; i < 3; i++)
        {
            src_v_temp[i] = src_v[i];
            trg_v_temp[i] = trg_v[i];
        }
        int i[3] = {0, 1, 2};
        int j[3] = {0, 1, 2};
        int src_int[3] = {src.indices.x, src.indices.y, src.indices.z};
        int trg_int[3] = {trg.indices.x, trg.indices.y, trg.indices.z};
        if (neighbor_num == 2)
        {
            int idx = 0;
            for (int jj = 0; jj < 3; jj++)
                for (int ii = 0; ii < 3; ii++)
                    if (src_int[ii] == trg_int[jj])
                    {
                        i[idx] = ii;
                        j[idx] = jj;
                        idx++;
                    }
            i[2] = 3 - i[0] - i[1];
            j[2] = 3 - j[0] - j[1];
        }
        if (neighbor_num == 1)
        {
            for (int ii = 0; ii < 3; ii++)
                for (int jj = 0; jj < 3; jj++)
                    if (src_int[ii] == trg_int[jj])
                    {
                        if (ii != 0)
                        {
                            i[0] = ii;
                            i[ii] = 0;
                        }
                        if (jj != 0)
                        {
                            j[0] = jj;
                            j[jj] = 0;
                        }
                    }
        }
        for (int idx = 0; idx < 3; idx++)
        {
            src_v[idx] = src_v_temp[i[idx]];
            trg_v[idx] = trg_v_temp[j[idx]];
        }
        return singular_integrand(src_v, trg_v, src_jacobian, trg_jacobian, src_norm, k, neighbor_num, type);
    }
}

template <class Tri>
inline __device__ __host__ complex face2PointIntegrand(const Tri &src, float3 trg, complex k, PotentialType type)
{
    float3 src_v[3] = {{src.v0}, {src.v1}, {src.v2}};
    float src_jacobian = jacobian(src_v);
    return potential_integrand(trg, src_v, src_jacobian, k, type);
}

NWOB_NAMESPACE_END