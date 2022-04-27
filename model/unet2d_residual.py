import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.project =nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += self.project(residual)
        return F.relu(x)


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()

        self.block = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
        # self.pool = nn.Conv2d(out_ch, out_ch, 3,stride=2,padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.upsample(x)
        x = self.block(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class ResidualUNet2D(nn.Module):

    def __init__(self, n_channels,nfeatures,n_emb=16,n_boundary=2):
        super(ResidualUNet2D, self).__init__()

        self.inconv = InConv(n_channels, nfeatures[0])
        self.down1 = Down(nfeatures[0], nfeatures[1])
        self.down2 = Down(nfeatures[1], nfeatures[2])
        self.down3 = Down(nfeatures[2], nfeatures[3])
        self.down4 = Down(nfeatures[3], nfeatures[4])


        # self.down5 = Down(nfeatures[4], nfeatures[5])
        self.up1_spix = Up(nfeatures[4], nfeatures[4])
        self.up2_spix = Up(nfeatures[4]+nfeatures[3], nfeatures[3])
        self.up3_spix = Up(nfeatures[3]+nfeatures[2], nfeatures[2])
        self.up4_spix = Up(nfeatures[2]+nfeatures[1], nfeatures[1])
        self.outconv_spix = OutConv(nfeatures[1], n_boundary)

        self.up1_emb = Up(nfeatures[4], nfeatures[4])
        self.up2_emb = Up(nfeatures[4]+nfeatures[3], nfeatures[3])
        self.up3_emb = Up(nfeatures[3]+nfeatures[2], nfeatures[2])
        self.up4_emb = Up(nfeatures[2]+nfeatures[1], nfeatures[1])
        self.outconv_emb = OutConv(nfeatures[1], n_emb)

        self.softmax = nn.Softmax(1)

        self.binary_seg = nn.Sequential(
            nn.Conv2d(nfeatures[1], nfeatures[1], 1),
            nn.BatchNorm2d(nfeatures[1]),
            nn.ReLU(),
            nn.Conv2d(nfeatures[1], 2, 1)
        )
    def concat_channels(self, x_cur, x_prev):
        if x_cur.shape!=x_prev.shape:
            p1 = x_prev.shape[-1]-x_cur.shape[-1]
            p2 = x_prev.shape[-2] - x_cur.shape[-2]
            padding = nn.ReplicationPad2d((0, p1, 0, p2)).cuda()
            x_cur = padding(x_cur)
        return torch.cat([x_cur, x_prev], dim=1)

    def forward(self, x):
        #encoder
        x = self.inconv(x)
        x2 = self.down1(x)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        # x = self.down5(x5)

        #superpixel
        x_spix = self.up1_spix(x)
        x_spix = self.concat_channels(x_spix, x4)
        x_spix = self.up2_spix(x_spix)
        x_spix = self.concat_channels(x_spix, x3)
        x_spix = self.up3_spix(x_spix)
        x_spix = self.concat_channels(x_spix, x2)
        x_spix = self.up4_spix(x_spix)
        spix4 = x_spix
        x_spix = self.outconv_spix(x_spix)


        #embedding
        x_emb0 = self.up1_emb(x)
        x_emb0 = self.concat_channels(x_emb0, x4)
        x_emb0 = self.up2_emb(x_emb0)
        x_emb0 = self.concat_channels(x_emb0, x3)
        x_emb0 = self.up3_emb(x_emb0)
        x_emb0 = self.concat_channels(x_emb0, x2)
        x_emb0 = self.up4_emb(x_emb0)
        x_emb = self.outconv_emb(x_emb0)
        binary_seg = self.binary_seg(x_emb0)
        # x_emb = self.softmax(x_emb)

        return x_spix,x_emb,binary_seg,spix4


if __name__ == '__main__':
    import numpy as np

    x = torch.Tensor(np.random.random((1, 1, 256, 256)).astype(np.float32)).cuda()

    # x = torch.Tensor(np.random.random((1, 1, 100, 256, 256)).astype(np.float32)).cuda()
    # mask = torch.Tensor(np.random.random((1, 1, 100, 256, 256)).astype(np.float32)).cuda()

    # x = torch.Tensor(np.random.random((1, 1, 66, 320, 320)).astype(np.float32)).cuda()
    # mask = torch.Tensor(np.random.random((1, 1, 66, 320, 320)).astype(np.float32)).cuda()

    # model = RSUNet_Nested([16,32,48,64,80]).cuda()
    # model = ResidualUNet2D(1,[3, 16, 32, 64,128])
    model = eval('ResidualUNet2D(1,[16,32,64,128,256])').cuda()

    # with torch.no_grad():
    out_o, out_c,out_s = model(x)
    print(out_o.shape, out_c.shape,out_s.shape)