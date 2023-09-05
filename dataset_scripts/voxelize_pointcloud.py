import sys
sys.path.append('..')
from src.classic.tools import *
from glob import glob
from tqdm import tqdm
import os

def dir(file_name):
    dir_name = os.path.dirname(file_name)
    os.makedirs(dir_name, exist_ok=True)

if __name__ == '__main__':
    file_list = glob(sys.argv[1])
    out_dir = sys.argv[2]

    print('file_number:', len(file_list))
    for filename in tqdm(file_list):
        try:
            voxels = voxelize_pointcloud(filename, augnum = 0)
        except:
            voxels = None

        if voxels is None:
            continue

        for i, voxel in enumerate(voxels):
            if voxel is None:
                continue
            out_file_name = os.path.join(out_dir, os.path.basename(filename).replace('.obj', str(i) + '.npy'))
            dir(out_file_name)
            np.save(out_file_name, voxel)



