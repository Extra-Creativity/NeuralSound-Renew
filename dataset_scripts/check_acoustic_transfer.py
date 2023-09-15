import sys

sys.path.append("./")
sys.path.append("../")
import numpy as np
from glob import glob
from src.classic.fem.project_util import voxel_to_edge, vert2vox
from src.classic.fem.util import to_sparse_coords
from src.classic.fem.assembler import vertices
from src.visualize import get_figure
import torch

data_file = sys.argv[1]
data = torch.load(data_file)
x = data["x"]
voxel = data["voxel"].numpy()

coords = to_sparse_coords(voxel)
vecs = x[:, voxel == 1].T.numpy()
mode_num = vecs.shape[-1]
get_figure(coords, vecs).show()
