import sys
import argparse
sys.path.append('..')
from src.net.unet import UNet3D
from src.classic.tools import index2mel, mel2val
import torch
from tqdm import tqdm
from glob import glob
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--statedir', type=str)
parser.add_argument('--datadir', type=str)
args = parser.parse_args()
 
model = UNet3D(1, 32, args.filter_num).cuda()
model.load_state_dict(torch.load(args.statedir))
model.eval()

for path in tqdm(glob(args.datadir)):
    voxel = torch.from_numpy(np.load(path))
    x = voxel.unsqueeze(0).bool()

    y_pred0, mask_pred, amp_pred = model(x)
    y_pred = y_pred0 * x
    y = y[mask_pred]

    mel = index2mel(np.arange(mask_pred.shape))
    vals = mel2val(mel)

    out_path = os.path.basename(path) + ".npz"
    np.savez(out_path, vecs = y, vals = vals)