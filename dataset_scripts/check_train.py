import sys

sys.path.append("./")
sys.path.append("../")
from src.classic.fem.util import to_sparse_coords
from src.visualize import get_figure
import torch

data_file = sys.argv[1]
data = torch.load(data_file)
x = data["x"]
if x.is_cuda:
    x = x.cpu().detach()
    # y = data['y'].cpu().detach()
    y_pred_x = data["y_pred_x"].cpu().detach()
else:
    # y = data['y']
    y_pred_x = data["y_pred_x"]

print(x.shape, y_pred_x.shape)

voxel = x[0, 0].numpy()
coords = to_sparse_coords(voxel)

vecs = y_pred_x[0, :, voxel == 1].T.numpy()
get_figure(coords, vecs).show()

# vecs = y[0, :, voxel == 1].T.numpy()
# get_figure(coords, vecs).show()
