import sys

sys.path.append("./")
sys.path.append("../")
import torch
from src.visualize import get_figure

data_file = sys.argv[1]
data = torch.load(data_file)
vertices = data["vertices"]
triangles = data["triangles"]
neumann = data["neumann"]
centers = vertices[triangles].mean(1).cpu().numpy()
neumann = neumann.unsqueeze(1).cpu().numpy()
print(centers.shape, neumann.shape)
get_figure(centers, neumann).show()
