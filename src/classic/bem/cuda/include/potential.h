
#pragma once
#include "helper_math.h"
#include "common.h"
NWOB_NAMESPACE_BEGIN
enum PotentialType
{
    SINGLE_LAYER,
    DOUBLE_LAYER
};
__device__ __host__ inline complex single_layer_potential(float3 src_coord, float3 trg_coord, complex k)
{
    complex potential(0, 0);
    float3 s2t = trg_coord - src_coord;
    float r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
    if (r2 != 0)
    {
        float r = sqrt(r2);
        potential += exp(complex(0, 1) * r * k) / (4 * M_PI * r);
        // printf("r = %f, k = %f, potential = %f + %f i\n", r, k.real(), potential.real(), potential.imag());
    }
    return potential;
}

__device__ __host__ inline complex double_layer_potential(float3 src_coord,
                                                          float3 trg_coord,
                                                          float3 trial_norm,
                                                          complex k)
{
    complex potential(0, 0);
    float3 s2t = trg_coord - src_coord;
    float r2 = s2t.x * s2t.x + s2t.y * s2t.y + s2t.z * s2t.z;
    if (r2 != 0)
    {
        float r = sqrt(r2);
        complex ikr = complex(0, 1) * r * k;
        potential += -exp(ikr) / (4 * M_PI * r2 * r) * (1 - ikr) * dot(s2t, trial_norm);
        // printf("exp(ikr) = %e + %e i, dot(s2t, trial_norm) = %e, potential = %e + %e i\n", exp(ikr).real(),
        //        exp(ikr).imag(), dot(s2t, trial_norm), potential.real(), potential.imag());
    }

    return potential;
}
__device__ __host__ inline complex layer_potential(float3 src_coord,
                                                   float3 trg_coord,
                                                   float3 trial_norm,
                                                   complex k,
                                                   PotentialType type)
{
    if (type == SINGLE_LAYER)
        return single_layer_potential(src_coord, trg_coord, k);
    else
        return double_layer_potential(src_coord, trg_coord, trial_norm, k);
}

NWOB_NAMESPACE_END
