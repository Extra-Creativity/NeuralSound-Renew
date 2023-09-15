import torch
import torch.nn.functional as F


def voxel_to_edge(voxel, device='cuda'):
    '''
    input:
        voxel: [batch_size, 1, res, res, res]
    return:
        hyper_edge: [edge_num, 8]
    '''
    device = torch.device(device)
    voxel = torch.from_numpy(voxel).unsqueeze(
        0).unsqueeze(0).to(device).float()
    grid = F.conv_transpose3d(voxel, torch.ones(
        1, 1, 2, 2, 2, device=voxel.device).float())
    mask = (grid > 0)
    grid[mask] = torch.arange(mask.sum(), device=voxel.device).float()
    kernel = torch.tensor([
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0, 0,
    ]).reshape(8, 1, 2, 2, 2).float().to(voxel.device)
    hyper_edge = F.conv3d(grid, kernel).permute(0, 2, 3, 4, 1)
    return hyper_edge[voxel.squeeze(1) > 0].long().cpu().numpy()


def vert2vox(x, edge):
    '''
    input:
        x:      [vert_num, feature_num]
        edge:   [voxel_num, 8]
    return:
        x_:     [voxel_num, 8*feature_num]
    '''
    voxel_num = edge.shape[0]
    feature_num = x.shape[-1]
    return x[edge].reshape(voxel_num, 8*feature_num)
