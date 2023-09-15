import sys

sys.path.append("./")
sys.path.append("../")
from src.classic.bem.assemble import (
    assemble_double_boundary_matrix,
    assemble_single_boundary_matrix,
    assemble_double_potential_matrix,
    assemble_single_potential_matrix,
    assemble_identity_matrix,
)
import bempp.api
import torch
import time

bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"

import numpy as np

sample_test = False

if sample_test:
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [2, 1, 1],
            [1, 1, 2],
        ]
    )
    triangles = np.array([[0, 1, 2]])
    grid = bempp.api.Grid(vertices.T, triangles.T)
else:
    grid = bempp.api.shapes.sphere(h=0.3)

vertices = torch.from_numpy(grid.vertices.T.astype("float32")).cuda()
triangles = torch.from_numpy(grid.elements.T.astype("int32")).cuda()

space = bempp.api.function_space(grid, "DP", 0)

for wave_number in [1, 10]:
    print(wave_number)
    start_time = time.time()
    slp = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, wave_number, device_interface="opencl", precision="single"
    )
    single_matrix_bempp = torch.from_numpy(slp.weak_form().A).cuda().real
    t1 = time.time() - start_time
    start_time = time.time()
    single_matrix_cuda = assemble_single_boundary_matrix(
        vertices, triangles, wave_number
    )
    torch.cuda.synchronize()
    t2 = time.time() - start_time
    error = abs(single_matrix_cuda - single_matrix_bempp) / (
        (single_matrix_bempp**2).mean() ** 0.5
    )
    print(
        "single max error: ",
        error.max(),
        " time cost bempp: ",
        t1,
        "time cost cuda: ",
        t2,
    )

    start_time = time.time()
    dlp = bempp.api.operators.boundary.helmholtz.double_layer(
        space, space, space, wave_number, device_interface="opencl", precision="single"
    )
    double_matrix_bempp = torch.from_numpy(dlp.weak_form().A).cuda().real
    t1 = time.time() - start_time
    start_time = time.time()
    double_matrix_cuda = assemble_double_boundary_matrix(
        vertices, triangles, wave_number
    )
    t2 = time.time() - start_time
    # print(double_matrix_cuda[:5, :5])
    # print(double_matrix_bempp[:5, :5])
    error = abs(double_matrix_cuda - double_matrix_bempp) / (
        (double_matrix_bempp**2).mean() ** 0.5
    )
    print(
        "double max error: ",
        error.max(),
        " time cost bempp: ",
        t1,
        "time cost cuda: ",
        t2,
    )
    eval_points = np.array(
        [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [3, 3, 3],
        ]
    )
    single_potential = bempp.api.operators.potential.helmholtz.single_layer(
        space, eval_points.T, wave_number, device_interface="opencl", precision="single"
    )

    coeff = np.random.rand(triangles.shape[0]).astype("float32")
    grid_fun = bempp.api.GridFunction(space, coefficients=coeff)

    bempp_result = single_potential.evaluate(grid_fun).real
    bempp_result = torch.from_numpy(bempp_result).cuda()
    cuda_mat = assemble_single_potential_matrix(
        vertices,
        triangles,
        torch.from_numpy(eval_points).cuda().float(),
        wave_number,
    )
    cuda_result = cuda_mat @ torch.from_numpy(coeff).cuda()
    error = abs(cuda_result - bempp_result) / ((bempp_result**2).mean() ** 0.5)
    print("single potential max error: ", error.max())

    double_potential = bempp.api.operators.potential.helmholtz.double_layer(
        space, eval_points.T, wave_number, device_interface="opencl", precision="single"
    )
    bempp_result = double_potential.evaluate(grid_fun).real
    bempp_result = torch.from_numpy(bempp_result).cuda()
    cuda_mat = assemble_double_potential_matrix(
        vertices,
        triangles,
        torch.from_numpy(eval_points).cuda().float(),
        wave_number,
    )
    cuda_result = cuda_mat @ torch.from_numpy(coeff).cuda()
    error = abs(cuda_result - bempp_result) / ((bempp_result**2).mean() ** 0.5)
    print("double potential max error: ", error.max())


identity = bempp.api.operators.boundary.sparse.identity(
    space, space, space, device_interface="opencl", precision="single"
)

identity_matrix_bempp = identity.weak_form().A.todense()
identity_matrix_bempp = torch.from_numpy(identity_matrix_bempp).cuda()

identity_matrix_cuda = assemble_identity_matrix(vertices, triangles)

error = abs(identity_matrix_cuda - identity_matrix_bempp) / (
    (identity_matrix_bempp**2).mean() ** 0.5
)
print("identity max error: ", error.max())
