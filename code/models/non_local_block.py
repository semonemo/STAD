import torch
from torch import nn
from torch.nn import functional as F


# embedded gaussian version
# https://github.com/AlexHex7/Non-local_pytorch
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, ref=False):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.ref = ref

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            conv_ref = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d

            if self.ref:
                conv_ref = nn.Conv2d
                bn = nn.BatchNorm2d

        elif dimension == 2:
            conv_nd = nn.Conv2d
            conv_ref = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            raise NotImplementedError

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_ref(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_ref(in_channels=self.inter_channels, out_channels=self.in_channels,
                              kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
        #                      kernel_size=1, stride=1, padding=0)
        self.theta = conv_ref(in_channels=self.in_channels, out_channels=self.inter_channels,
                              kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, 2c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (b, c, thw)
        g_x = g_x.permute(0, 2, 1)  # (b, thw, c)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # (b, thw, c)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (b, c, thw)
        f = torch.matmul(theta_x, phi_x)  # (b, thw, thw)
        f_div_C = F.softmax(f, dim=-1)  # (b, thw, thw)

        yy = torch.matmul(f_div_C, g_x)  # (b, thw, c)
        yy = yy.permute(0, 2, 1).contiguous()
        yy = yy.view(batch_size, self.inter_channels, *x.size()[2:])  # (b, c, t, h, w)
        W_y = self.W(yy)  # (b, 2c, t, h, w)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

    def forward_ref(self, x, ref_idx=-1, return_nl_map=False):
        """
        :param x: (b, 2c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        batch_size = x.size(0)
        x_ref = x[:, :, ref_idx]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (b, c, thw)
        g_x = g_x.permute(0, 2, 1)  # (b, thw, c)
        theta_x = self.theta(x_ref).view(batch_size, self.inter_channels, -1)  # only consider reference feat
        theta_x = theta_x.permute(0, 2, 1)  # (b, hw, c)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (b, c, thw)
        f = torch.matmul(theta_x, phi_x)  # (b, hw, thw)
        f_div_C = F.softmax(f, dim=-1)  # (b, hw, thw)

        yy = torch.matmul(f_div_C, g_x)  # (b, hw, c)
        yy = yy.permute(0, 2, 1).contiguous()
        yy = yy.view(batch_size, self.inter_channels, *x.size()[-2:])  # (b, c, h, w)
        W_y = self.W(yy)  # (b, 2c, h, w)

        z = W_y + x_ref
        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, )


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, ref=False):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, ref=ref)


if __name__ == '__main__':
    import torch

    img = torch.randn(2, 14, 20, 20)  # (b, c, h, w)
    net = NONLocalBlock2D(img.shape[1], sub_sample=False, bn_layer=True)
    out = net(img)
    print(out.size())

    img = torch.randn(2, 14, 8, 20, 20)  # (b, c, t, h, w)
    net = NONLocalBlock3D(img.shape[1], sub_sample=False, bn_layer=True)
    out = net(img)
    print(out.size())
