import sys

sys.path.append("./")
sys.path.append("../")
import numpy as np
from glob import glob
from src.classic.fem.project_util import voxel_to_edge, vert2vox
from src.classic.fem.util import to_sparse_coords
from src.classic.fem.assembler import vertices
from src.visualize import get_figure

data_file = sys.argv[1]
data = np.load(data_file)
voxel = data["voxel"]
vecs = data["vecs"]
mode_num = vecs.shape[-1]
vecs = vecs.reshape(-1, 3, mode_num)
vecs = (vecs**2).sum(axis=1) ** 0.5

coords = to_sparse_coords(voxel)
edge = voxel_to_edge(voxel)
vecs_vox = vert2vox(vecs, edge)

get_figure(coords, vecs_vox).show()
