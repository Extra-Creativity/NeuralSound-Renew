from time import time
from .bemModel import boundary_mesh
from .util import voxel2boundary, unit_sphere_surface_points, obj_to_grid
import numpy as np
import os
from tqdm import tqdm
import torch

SPEED_OF_SOUND = 343
AIR_DENSITY = 1.225


def potential_compute(data_list, points_list, image_size):
    data_sum = 0
    for data, points in zip(data_list, points_list):
        points = points.reshape(-1, 3)
        r = (points**2).sum(-1) ** 0.5
        data = torch.abs(data) * r
        data_sum = data + data_sum
    data = data_sum / len(data_list)
    return data.reshape(2 * image_size, image_size)


def vibration_to_ffat(
    voxel_coords, voxel_vib, freqs, length=0.15, image_size=32, debug=False
):
    voxel_res = 32
    vertices, elements, feats_index = map(
        np.asarray, voxel2boundary(voxel_coords, voxel_res)
    )
    vertices = (vertices / voxel_res - 0.5) * length
    vertices = torch.from_numpy(vertices).float().cuda()
    triangles = torch.from_numpy(elements).long().cuda()

    voxel_num = len(voxel_coords)
    map_num = len(freqs)
    ffat_map = torch.zeros((map_num, 2 * image_size, image_size))
    ffat_map_far = torch.zeros((map_num, 2 * image_size, image_size))
    feats_in_reshaped = (voxel_vib.reshape(voxel_num, 3, -1) ** 2).sum(1) ** 0.5

    points_ = unit_sphere_surface_points(image_size) * length
    points_ = torch.from_numpy(points_).float().cuda()
    points_list = [points_ * 1.25, points_ * 1.25**2, points_ * 1.25**3]
    points_list_far = [points_ * 3, points_ * 3**2, points_ * 3**3]

    for i in range(map_num):
        feats_select = feats_in_reshaped[:, i]
        omega = 2 * np.pi * freqs[i]
        wave_number = omega / SPEED_OF_SOUND
        neumann = feats_select[feats_index]
        neumann = torch.from_numpy(neumann).float().cuda()
        bm = boundary_mesh(vertices, triangles, wave_number)
        dirichlet = bm.solve_dirichlet(neumann)
        data_list = [
            bm.solve_points(ps.reshape(-1, 3), neumann, dirichlet) for ps in points_list
        ]
        ffat_map[i] = potential_compute(data_list, points_list, image_size)
        data_list = [
            bm.solve_points(ps.reshape(-1, 3), neumann, dirichlet)
            for ps in points_list_far
        ]
        ffat_map_far[i] = potential_compute(data_list, points_list_far, image_size)

        if debug:
            torch.save(
                {
                    "vertices": vertices,
                    "triangles": triangles,
                    "neumann": neumann,
                },
                f"output/debug_{i}.pt",
            )
    return ffat_map, ffat_map_far


def ffat_to_imgs(ffat_map, output_dir, tag=""):
    """
    ffat_map: [n, size1, size2]
    """
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)
    for idx, data in enumerate(ffat_map):
        data = data / data.max() * 255
        Image.fromarray(np.uint8(data), "L").save(f"{output_dir}/{tag}{idx}.jpg")
