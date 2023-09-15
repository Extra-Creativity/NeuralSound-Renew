import sys

sys.path.append("./")
sys.path.append("../")
from src.classic.tools import *
from glob import glob
from tqdm import tqdm
import os


def dir(file_name):
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)


if __name__ == "__main__":
    file_list = glob(sys.argv[1])
    out_dir = sys.argv[2]
    file_success_count = len(glob(out_dir + "/*"))
    for filename in tqdm(file_list):
        file_success_count -= 1
        if file_success_count >= 0:
            continue
        voxel = np.load(filename)
        # k is the number of modes
        try:
            vecs, vals = modal_analysis(voxel, k=20, mat=Material.Ceramic)
        except:
            continue
        out_file_name = os.path.join(
            out_dir, os.path.basename(filename).replace(".npy", ".npz")
        )
        dir(out_file_name)
        np.savez_compressed(out_file_name, voxel=voxel, vecs=vecs, vals=vals)
