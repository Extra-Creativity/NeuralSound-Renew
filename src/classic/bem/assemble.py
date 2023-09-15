import numpy as np
import torch
from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob


class CUDA_MODULE:
    _module = None

    @staticmethod
    def get(name):
        if CUDA_MODULE._module is None:
            CUDA_MODULE.load()
        return getattr(CUDA_MODULE._module, name)

    @staticmethod
    def load(Debug=False, MemoryCheck=False, Verbose=False):
        src_dir = os.path.dirname(os.path.abspath(__file__)) + "/cuda"
        os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(src_dir, "build")
        cflags = "--extended-lambda --expt-relaxed-constexpr "
        if Debug:
            cflags += " -G -g"
            cflags += " -DDEBUG"
        else:
            cflags += " -O3"
            cflags += " -DNDEBUG"
        if MemoryCheck:
            cflags += " -DMEMORY_CHECK"
        cuda_files = [os.path.join(src_dir, "bind.cu")]
        include_paths = [src_dir, os.path.join(src_dir, "include")]
        CUDA_MODULE._module = load_cuda(
            name="CUDA_MODULE",
            sources=cuda_files,
            extra_include_paths=include_paths,
            extra_cuda_cflags=[cflags],
            verbose=Verbose,
        )
        return CUDA_MODULE._module


CUDA_MODULE.load(Debug=False, MemoryCheck=False, Verbose=False)


def assemble_single_boundary_matrix(vertices, triangles, wave_number):
    vertices = vertices.to(torch.float32)
    triangles = triangles.to(torch.int32)
    matrix_cuda = torch.zeros(
        triangles.shape[0], triangles.shape[0], device="cuda", dtype=torch.float32
    )
    CUDA_MODULE.get("face2face")(vertices, triangles, matrix_cuda, wave_number, False)
    return matrix_cuda


def assemble_double_boundary_matrix(vertices, triangles, wave_number):
    vertices = vertices.to(torch.float32)
    triangles = triangles.to(torch.int32)
    matrix_cuda = torch.zeros(triangles.shape[0], triangles.shape[0], device="cuda")
    CUDA_MODULE.get("face2face")(vertices, triangles, matrix_cuda, wave_number, True)
    return matrix_cuda


def assemble_single_potential_matrix(vertices, triangles, points, wave_number):
    vertices = vertices.to(torch.float32)
    triangles = triangles.to(torch.int32)
    points = points.to(torch.float32)
    matrix_cuda = torch.zeros(triangles.shape[0], points.shape[0], device="cuda")
    CUDA_MODULE.get("face2points")(
        vertices, triangles, points, matrix_cuda, wave_number, False
    )
    return matrix_cuda.T


def assemble_double_potential_matrix(vertices, triangles, points, wave_number):
    vertices = vertices.to(torch.float32)
    triangles = triangles.to(torch.int32)
    points = points.to(torch.float32)
    matrix_cuda = torch.zeros(triangles.shape[0], points.shape[0], device="cuda")
    CUDA_MODULE.get("face2points")(
        vertices, triangles, points, matrix_cuda, wave_number, True
    )
    return matrix_cuda.T


def assemble_identity_matrix(vertices, triangles):
    vertices = vertices.to(torch.float32)
    triangles = triangles.to(torch.int32)
    diagnal = torch.zeros(triangles.shape[0], device="cuda", dtype=torch.float32)
    CUDA_MODULE.get("identity")(vertices, triangles, diagnal)
    return torch.diag(diagnal)
