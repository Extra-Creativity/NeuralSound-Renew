import sys

sys.path.append("./")
sys.path.append("../")
from src.classic.fem.util import to_sparse_coords, vertex_to_voxel_matrix
from src.classic.bem.ffat import ffat_to_imgs, vibration_to_ffat
from src.classic.bem.util import boundary_voxel
from src.classic.tools import val2freq
from src.classic.fem.femModel import Material, random_material
from tqdm import tqdm
import numpy as np
import os
from glob import glob
import torch


def process_single_model(filename, output_name):
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    data = np.load(filename)
    vecs, vals, voxel = data["vecs"], data["vals"], data["voxel"]
    freqs = val2freq(vals)
    coords = to_sparse_coords(voxel)
    coords_surface, _ = map(np.asarray, boundary_voxel(coords))
    # Skip objects with too many suface voxels for feasible time cost.
    if len(coords_surface) > 5500:
        return

    # Random material and object size for different frequency
    vertex_num = vecs.shape[0] // 3
    mode_num = len(freqs)
    idx_lst = np.arange(mode_num)
    np.random.shuffle(idx_lst)
    freqs_lst = []
    vecs_lst = []
    for mode_idx in idx_lst:
        vec = vertex_to_voxel_matrix(voxel) @ vecs[:, mode_idx].reshape(vertex_num, -1)
        length = np.random.rand() + 0.05
        freq = freqs[mode_idx] * Material.omega_rate(
            Material.Ceramic, random_material(), length / 0.15
        )
        if freq < 200:
            freq = 200
        if freq > 15000:
            freq = 15000
        freqs_lst.append(freq)
        vecs_lst.append(vec)
    vecs = np.stack(vecs_lst, axis=2)
    freqs = np.array(freqs_lst)

    # caculate ffat map
    ffat_map, ffat_map_far = vibration_to_ffat(coords, vecs, freqs)
    # save ffat map to image for visualization
    # ffat_to_imgs(ffat_map.cpu().numpy(), "./output", tag="ffat")
    # ffat_to_imgs(ffat_map_far.cpu().numpy(), "./output", tag="ffat_far")
    vecs = (vecs.reshape(-1, 3, mode_num) ** 2).sum(1) ** 0.5
    vecs = torch.from_numpy(vecs).float()
    voxel = torch.from_numpy(voxel).bool()
    x = torch.zeros(mode_num, *voxel.shape).float()
    x[:, voxel == 1] = vecs.T
    y0 = ffat_map
    y1 = ffat_map_far
    torch.save(
        {
            "voxel": voxel,
            "x": x,
            "y0": y0,
            "y1": y1,
        },
        output_name.replace(".npz", ".pt"),
    )


if __name__ == "__main__":
    file_list = glob(sys.argv[1])
    out_dir = sys.argv[2]
    for filename in tqdm(file_list):
        out_name = os.path.join(out_dir, os.path.basename(filename))
        print(out_name)
        process_single_model(filename, out_name)
