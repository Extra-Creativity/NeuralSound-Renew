# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,
    )


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1
        ),
        nn.BatchNorm3d(out_dim),
        activation,
    )


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )


class UNet3D(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet3D, self).__init__()

        self.in_dim = in_dim
        print(in_dim, out_dim, num_filters)
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(
            self.num_filters, self.num_filters * 2, activation
        )
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(
            self.num_filters * 2, self.num_filters * 4, activation
        )
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(
            self.num_filters * 4, self.num_filters * 8, activation
        )
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(
            self.num_filters * 8, self.num_filters * 16, activation
        )
        self.pool_5 = max_pooling_3d()

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.mid_mlp = nn.Sequential(
            nn.Linear(self.num_filters * 16, self.num_filters * 16),
            nn.BatchNorm1d(self.num_filters * 16),
            activation,
            nn.Linear(self.num_filters * 16, self.num_filters * 16),
            nn.BatchNorm1d(self.num_filters * 16),
            activation,
            nn.Linear(self.num_filters * 16, out_dim * 2),
        )

        # Bridge
        self.bridge = conv_block_2_3d(
            self.num_filters * 16, self.num_filters * 32, activation
        )

        # Up sampling
        self.trans_1 = conv_trans_block_3d(
            self.num_filters * 32, self.num_filters * 32, activation
        )
        self.up_1 = conv_block_2_3d(
            self.num_filters * 48, self.num_filters * 16, activation
        )
        self.trans_2 = conv_trans_block_3d(
            self.num_filters * 16, self.num_filters * 16, activation
        )
        self.up_2 = conv_block_2_3d(
            self.num_filters * 24, self.num_filters * 8, activation
        )
        self.trans_3 = conv_trans_block_3d(
            self.num_filters * 8, self.num_filters * 8, activation
        )
        self.up_3 = conv_block_2_3d(
            self.num_filters * 12, self.num_filters * 4, activation
        )
        self.trans_4 = conv_trans_block_3d(
            self.num_filters * 4, self.num_filters * 4, activation
        )
        self.up_4 = conv_block_2_3d(
            self.num_filters * 6, self.num_filters * 2, activation
        )
        self.trans_5 = conv_trans_block_3d(
            self.num_filters * 2, self.num_filters * 2, activation
        )
        self.up_5 = conv_block_2_3d(
            self.num_filters * 3, self.num_filters * 1, activation
        )

        # Output
        self.out = nn.Conv3d(num_filters, out_dim, kernel_size=1)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)  # -> [1, 4, 32, 32, 32]
        pool_1 = self.pool_1(down_1)  # -> [1, 4, 16, 16, 16]

        down_2 = self.down_2(pool_1)  # -> [1, 8, 16, 16, 16]
        pool_2 = self.pool_2(down_2)  # -> [1, 8, 8, 8, 8]

        down_3 = self.down_3(pool_2)  # -> [1, 16, 8, 8, 8]
        pool_3 = self.pool_3(down_3)  # -> [1, 4, 4, 4, 16]

        down_4 = self.down_4(pool_3)  # -> [1, 32, 4, 4, 4]
        pool_4 = self.pool_4(down_4)  # -> [1, 32, 2, 2, 2]

        down_5 = self.down_5(pool_4)  # -> [1, 64, 2, 2, 2]
        pool_5 = self.pool_5(down_5)  # -> [1, 64, 1, 1, 1]

        # Bridge
        bridge = self.bridge(pool_5)  # -> [1, 128, 1, 1, 1]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 2, 2, 2]
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> [1, 192, 2, 2, 2]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 2, 2, 2]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 4, 4, 4]
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> [1, 96, 4, 4, 4]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 4, 4, 4]

        trans_3 = self.trans_3(up_2)  # -> [1, 8, 8, 8, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> [1, 48, 8, 8, 8]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 8, 8, 8]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 16, 16, 16]
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> [1, 24, 16, 16, 16]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 16, 16, 16]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 32, 32, 32]
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> [1, 12, 32, 32, 32]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 32, 32, 32]

        # Output
        out = self.out(up_5)  # -> [1, 1, 32, 32, 32]

        mid_data = self.global_pool(pool_5).squeeze()
        mid_data = self.mid_mlp(mid_data)
        amp = mid_data[:, : self.out_dim]
        mask = mid_data[:, self.out_dim :]
        return out, mask, amp
