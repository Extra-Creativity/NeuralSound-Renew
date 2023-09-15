import numpy as np
import os
import open3d as o3d
from .fem.femModel import Hexahedron_model, Material
from .fem.solver import LOBPCG_solver, Lanczos_Solver
from .fem.util import to_sparse_coords
# from .bem.ffat import vibration_to_ffat
from .mesh.loader import ObjLoader

class MelConfig():
    mel_min, mel_max = 100, 10000
    mel_res = 32
    mel_spacing = (mel_max - mel_min) / mel_res

def mel2norm(mel):
    return (mel - MelConfig.mel_min) / (MelConfig.mel_max - MelConfig.mel_min)

def norm2mel(n):
    return n*(MelConfig.mel_max - MelConfig.mel_min) + MelConfig.mel_min

def mel_index(mel):
    idx = int((mel- MelConfig.mel_min)/MelConfig.mel_spacing)
    if idx >= MelConfig.mel_res:
        idx = MelConfig.mel_res - 1
    if idx <= 0:
        idx = 0
    return idx

def index2mel(idx):
    return MelConfig.mel_min + (idx + 0.5)*MelConfig.mel_spacing

def mel2freq(mel):
    return 700*(10**(mel/2595) - 1)

def freq2mel(f):
    return np.log10(f/700 + 1)*2595

def freq2val(f):
    return (f*(2*np.pi))**2

def val2freq(v):
    return v**0.5/(2*np.pi)

def val2mel(v):
    return freq2mel(val2freq(v))

def mel2val(mel):
    return freq2val(mel2freq(mel))
    
def voxelize_mesh(mesh_name, res = 32):
    from .voxelize.hexahedral import Hexa_model
    mesh = ObjLoader(mesh_name)
    if mesh.vertices.shape[0] == 0:
        return None
    mesh.normalize()
    hexa = Hexa_model(mesh.vertices, mesh.faces, res=res)
    hexa.fill_shell()
    return hexa.voxel_grid

def voxelize_mesh_robust(mesh_name, res = 32):
    from .voxelize.hexahedral import Hexa_model
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    triangles = np.asarray(mesh.triangles)
    if triangles.shape[0] == 0:
        return None
    def normalize_mesh(vertices):
        max_bound, min_bound = vertices.max(0), vertices.min(0)
        vertices = (vertices - (max_bound+min_bound)/2) / (max_bound - min_bound).max()
        return vertices

    vertices = np.asarray(mesh.vertices)
    vertices = normalize_mesh(vertices)
    hexa = Hexa_model(vertices, triangles, res)
    hexa.fill_shell()
    return hexa.voxel_grid

def voxelize_pointcloud(mesh_name, res = 32, augnum = 0):
    from .voxelize.hexahedral import Hexa_model
    
    mesh = o3d.io.read_triangle_mesh(mesh_name)
    tri_num = np.asarray(mesh.triangles).shape[0]
    if tri_num == 0:
        return None
    
    def normalize_pointcloud(point_cloud):
        points = np.asarray(point_cloud.points)
        max_bound, min_bound = points.max(0), points.min(0)
        points = (points - (max_bound+min_bound)/2) / (max_bound - min_bound).max()
        return points

    point_num = tri_num * 4
    point_clouds = [mesh.sample_points_uniformly(number_of_points=point_num)]
    results = []
    for _ in range(augnum):
        rotateAngles = np.random.uniform(-np.pi, np.pi, (3, 1))
        rotateMat = o3d.geometry.get_rotation_matrix_from_xyz(rotateAngles)
        newMesh = mesh.rotate(rotateMat)
        point_clouds.append(newMesh.sample_points_uniformly(number_of_points=point_num))

    for point_cloud in point_clouds:
        points = normalize_pointcloud(point_cloud)
        hexa = Hexa_model(points, None, res)
        hexa.fill_shell()
        results.append(hexa.voxel_grid)

    return results

def modal_analysis(voxel, mat = Material.Ceramic, k = 20):
    model = Hexahedron_model(voxel,mat=mat)
    
    model.modal_analysis(LOBPCG_solver(k=k))
    return model.vecs, model.vals




