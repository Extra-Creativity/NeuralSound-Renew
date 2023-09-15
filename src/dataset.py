import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob


class DeepModalDataset(Dataset):
    def __init__(self, root_dir, phase):
        self.file_list = glob(root_dir + "/" + phase + "/*.pt")
        print(
            "Load %d data from %s"
            % (len(self.file_list), os.path.join(root_dir, phase))
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = torch.load(self.file_list[index])
        x = data["x"].float()
        y_ = data["y"].float()
        amp_ = data["amp"].float()
        mask = data["mask"]
        y = torch.zeros(len(mask), *x.shape[1:]).float()
        y[mask, :] = y_
        amp = torch.zeros_like(mask).float()
        amp[mask] = amp_
        mask_mask = torch.zeros_like(mask).bool()
        mask_index = torch.arange(mask.nonzero(as_tuple=True)[0].max() + 1)
        mask_mask[mask_index] = True
        """ 
        mask_mask is a mask of mask as 16 eigenvalues can not cover all the frequency bins,
        so we only train on the bins that are covered by the 16 eigenvalues (discard uncovered high frequency bins)
        """
        return x, y, mask, amp, mask_mask
