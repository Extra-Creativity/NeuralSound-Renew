import sys

sys.path.append("./")
sys.path.append("../")
from src.classic.tools import *
from src.classic.fem.project_util import voxel_to_edge, vert2vox
from src.classic.fem.util import to_sparse_coords
from src.classic.fem.assembler import vertices
from glob import glob
from tqdm import tqdm
import os
import torch


def process_single(vecs, vals, voxel, out_file_name):
    dir(out_file_name)
    mel_num = MelConfig.mel_res
    band_vecs = np.zeros((mel_num, vecs.shape[0]))
    mask = np.zeros(mel_num).astype(bool)

    mode_num = len(vals)
    for i in range(mode_num):
        mel_i = val2mel(vals[i])
        mel_idx = mel_index(mel_i)
        band_vecs[mel_idx] += vecs[:, i]
        mask[mel_idx] = True

    band_vecs = band_vecs[mask]
    band_amp = np.zeros(band_vecs.shape[0])
    for i in range(len(band_vecs)):
        band_amp[i] = (band_vecs[i] ** 2).mean() ** 0.5
        band_vecs[i] = band_vecs[i] / band_amp[i]

    band_vecs = torch.from_numpy(band_vecs).float()
    band_amp = torch.from_numpy(band_amp)
    mask = torch.from_numpy(mask)
    voxel = torch.from_numpy(voxel)

    y = torch.zeros(len(band_vecs), *voxel.shape).float()
    y[:, voxel == 1] = band_vecs
    x = voxel.unsqueeze(0).bool()
    torch.save(
        {
            "x": x,
            "y": y,
            "amp": band_amp,
            "mask": mask,
        },
        out_file_name,
    )


def dir(file_name):
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)


if __name__ == "__main__":
    file_list = sorted(glob(sys.argv[1]))
    out_dir = sys.argv[2]
    i, begin = 0, 0
    for filename in tqdm(file_list):
        if i < begin:
            i = i + 1
            continue
        data = np.load(filename)
        vecs, vals, voxel = data["vecs"], data["vals"], data["voxel"]
        mode_num = vecs.shape[-1]
        vecs = vecs.reshape(-1, 3, mode_num)
        vecs = (vecs**2).sum(axis=1) ** 0.5
        edge = voxel_to_edge(voxel)
        vecs_vox = vert2vox(vecs, edge)
        out_file_name = os.path.join(
            out_dir, "train", os.path.basename(filename).replace(".npz", ".pt")
        )
        if np.random.rand() < 0.2:
            # 20% of the data is used for validation
            out_file_name = out_file_name.replace("train", "val")
        try:
            process_single(vecs_vox, vals, voxel, out_file_name)
        except:
            continue
