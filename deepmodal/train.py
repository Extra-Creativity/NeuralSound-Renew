import sys
import argparse

sys.path.append("..")
from src.dataset import DeepModalDataset
from src.net.unet import UNet3D
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="test")
parser.add_argument(
    "--dataset", type=str, default="/data2/NeuralSound/ModelNet/pointcloud/deepmodal"
)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--filter_num", type=int, default=16)
parser.add_argument("--epoch", type=int, default=200)
args = parser.parse_args()

import os

log_dir = "log/%s" % args.tag
if os.path.exists(log_dir):
    os.system("rm -r %s" % log_dir)

writer = SummaryWriter(log_dir)
train_dataset = DeepModalDataset(args.dataset, "train")
val_dataset = DeepModalDataset(args.dataset, "val")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
)

model = UNet3D(1, 32, args.filter_num).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=4, verbose=True
)
loss_func = torch.nn.MSELoss()
loss_mask_func = torch.nn.BCEWithLogitsLoss()

def step(outdir, train=True):
    if train:
        model.train()
        loader = train_loader
    else:
        model.eval()
        loader = val_loader
    loss_sum = 0
    for x, y, mask, amp, mask_mask in tqdm(loader):
        x = x.cuda()
        y = y.cuda()
        mask = mask.cuda()
        amp = amp.cuda()
        y_pred0, mask_pred, amp_pred = model(x)
        print(y_pred0.shape, mask_pred.shape, amp_pred.shape)
        y_pred = y_pred0 * x
        k1 = 0.1
        k2 = 1.0
        loss = (
            loss_func(y_pred[mask], y[mask])
            + k1 * loss_mask_func(mask_pred[mask_mask], mask.float()[mask_mask])
            + k2 * loss_func(amp_pred[mask], amp[mask])
        )
        print(loss.item())
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item()

    torch.save(
        {
            "x": x.detach().cpu(),
            "y_pred": y_pred0.detach().cpu(),
            "y_pred_x": y_pred.detach().cpu(),
            "y": y.detach().cpu(),
        },
        outdir + ".pt",
    )

    return loss_sum / len(loader)


best_loss = 1e10

for epoch in range(args.epoch):
    train_loss = step("log/train/" + str(epoch), train=True)
    writer.add_scalar("train_loss", train_loss, epoch)
    val_loss = step("log/valid/" + str(epoch), train=False)
    writer.add_scalar("val_loss", val_loss, epoch)
    scheduler.step(val_loss)
    print("Epoch %d, train loss %.5f, val loss %.5f" % (epoch, train_loss, val_loss))
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), log_dir + "/best.pth")

writer.close()
