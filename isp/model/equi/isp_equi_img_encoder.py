from escnn import gspaces
from escnn import nn
import torch


def conv2d(
    feat_type_in,
    feat_type_hid,
    kernel_size,
    stride=1,
    groups=1,
    dilation=1,
    initialize=False,
):
    return nn.R2Conv(
        feat_type_in,
        feat_type_hid,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=(kernel_size - 1) // 2,
        groups=groups,
        initialize=initialize,
    )


class EquiBottleNeck(torch.nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        N=8,
        initialize=True,
        stride=1,
        down_sample=None,
        dilation=1,
        kernel_size=3,
    ):
        super(EquiBottleNeck, self).__init__()
        r2_act = gspaces.rot2dOnR2(N=N)
        rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, [rep] * in_planes)
        feat_type_hid = nn.FieldType(r2_act, [rep] * out_planes)

        self.conv1 = nn.R2Conv(
            feat_type_in,
            feat_type_hid,
            kernel_size=1,
            stride=1,
            dilation=dilation,
            padding=0,
            initialize=initialize,
        )

        padding = (kernel_size - 1) // 2
        self.conv2 = nn.R2Conv(
            feat_type_hid,
            feat_type_hid,
            kernel_size=kernel_size,
            initialize=initialize,
            dilation=dilation,
            stride=stride,
            padding=padding,
        )
        self.conv3 = nn.R2Conv(
            feat_type_hid,
            feat_type_hid,
            kernel_size=1,
            stride=1,
            dilation=dilation,
            padding=0,
            initialize=initialize,
        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        out += identity
        out = self.relu(out)

        return out


class EquiResNet_50(torch.nn.Module):
    def __init__(
        self, block, layers, obs_channel: int = 3, N=8, n_out: int = 64, initialize=True
    ):
        super(EquiResNet_50, self).__init__()
        self.obs_channel = obs_channel
        self.N = N
        self.r2_act = gspaces.rot2dOnR2(N=N)
        self.repr = self.r2_act.regular_repr
        self.in_planes = n_out // N

        feat_type_in = nn.FieldType(
            self.r2_act, [self.r2_act.trivial_repr] * obs_channel
        )
        feat_type_hid = nn.FieldType(self.r2_act, [self.repr] * self.in_planes)
        # 76x76
        self.conv1 = nn.R2Conv(
            feat_type_in,
            feat_type_hid,
            kernel_size=5,
            stride=1,
            padding=0,
            initialize=initialize,
        )

        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.maxpool = nn.PointwiseMaxPool(
            feat_type_hid, kernel_size=2, stride=2, padding=0, ceil_mode=True
        )

        self.layer1 = self._make_layer(
            block, n_out // 8, layers[0], stride=1, N=N, initialize=initialize
        )
        self.layer2 = self._make_layer(
            block,
            n_out // 4,
            layers[1],
            stride=1,
            N=N,
            initialize=initialize,
            pool=True,
        )
        self.layer3 = self._make_layer(
            block,
            n_out // 2,
            layers[2],
            stride=1,
            N=N,
            initialize=initialize,
            pool=True,
        )
        self.layer4 = self._make_layer(
            block, n_out, layers[3], stride=1, N=N, initialize=initialize
        )

        self.out_layer = nn.R2Conv(
            nn.FieldType(self.r2_act, n_out * [self.r2_act.regular_repr]),
            nn.FieldType(self.r2_act, 8 * n_out * [self.r2_act.trivial_repr]),
            kernel_size=3,
            padding=0,
            initialize=initialize,
        )

    def _make_layer(
        self, block, out_planes, block_num, stride=1, N=8, initialize=True, pool=False
    ):
        down_sample = None
        if self.in_planes != out_planes:
            down_sample = nn.SequentialModule(
                nn.R2Conv(
                    nn.FieldType(self.r2_act, [self.repr] * self.in_planes),
                    nn.FieldType(self.r2_act, [self.repr] * out_planes),
                    kernel_size=1,
                    stride=stride,
                    initialize=initialize,
                ),
            )
        if pool:
            layers = [
                nn.PointwiseMaxPool(
                    nn.FieldType(self.r2_act, [self.repr] * self.in_planes),
                    kernel_size=2,
                    padding=0,
                    ceil_mode=True,
                )
            ]
        else:
            layers = []

        layers.append(
            block(
                self.in_planes,
                out_planes,
                stride=1,
                down_sample=down_sample,
                N=N,
                initialize=initialize,
            )
        )

        self.in_planes = out_planes

        for i in range(1, block_num):
            layers.append(
                block(
                    self.in_planes,
                    out_planes,
                    stride=1,
                    down_sample=None,
                    N=N,
                    initialize=initialize,
                )
            )

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(
                x,
                nn.FieldType(
                    self.r2_act, self.obs_channel * [self.r2_act.trivial_repr]
                ),
            )

        #  76x76
        x = self.conv1(x)
        x = self.relu(x)
        # 72x72
        x = self.maxpool(x)
        # 36x36
        x = self.layer1(x)
        # 36x36
        x = self.layer2(x)
        # 18x18
        x = self.layer3(x)
        # 9x9
        x = self.layer4(x)
        # 7x7
        out = self.out_layer(x)
        return out


class EquiResNet_50_1x1(torch.nn.Module):
    def __init__(
        self,
        block,
        layers,
        obs_channel: int = 3,
        N=8,
        n_out: int = 128,
        initialize=True,
    ):
        super(EquiResNet_50_1x1, self).__init__()
        self.obs_channel = obs_channel
        self.N = N
        self.r2_act = gspaces.rot2dOnR2(N=N)
        self.repr = self.r2_act.regular_repr
        self.in_planes = n_out // N

        feat_type_in = nn.FieldType(
            self.r2_act, [self.r2_act.trivial_repr] * obs_channel
        )
        feat_type_hid = nn.FieldType(self.r2_act, [self.repr] * self.in_planes)
        # 76x76
        self.conv1 = nn.R2Conv(
            feat_type_in,
            feat_type_hid,
            kernel_size=5,
            stride=1,
            padding=0,
            initialize=initialize,
        )

        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.maxpool = nn.PointwiseMaxPool(
            feat_type_hid, kernel_size=2, stride=2, padding=0, ceil_mode=True
        )

        self.layer1 = self._make_layer(
            block, n_out // 8, layers[0], stride=1, N=N, initialize=initialize
        )
        self.layer2 = self._make_layer(
            block,
            n_out // 4,
            layers[1],
            stride=1,
            N=N,
            initialize=initialize,
            pool=True,
        )
        self.layer3 = self._make_layer(
            block,
            n_out // 2,
            layers[2],
            stride=1,
            N=N,
            initialize=initialize,
            pool=True,
        )
        self.layer4 = self._make_layer(
            block,
            n_out,
            layers[3],
            stride=1,
            N=N,
            initialize=initialize,
            pool=True,
            k=3,
        )

        self.out_layer = nn.R2Conv(
            nn.FieldType(self.r2_act, n_out * [self.r2_act.regular_repr]),
            nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
            kernel_size=3,
            padding=0,
            initialize=initialize,
        )
        self.relu_out = nn.ReLU(
            nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]), inplace=True
        )

    def _make_layer(
        self,
        block,
        out_planes,
        block_num,
        stride=1,
        N=8,
        initialize=True,
        pool=False,
        k=2,
    ):
        down_sample = None
        if self.in_planes != out_planes:
            down_sample = nn.SequentialModule(
                nn.R2Conv(
                    nn.FieldType(self.r2_act, [self.repr] * self.in_planes),
                    nn.FieldType(self.r2_act, [self.repr] * out_planes),
                    kernel_size=1,
                    stride=stride,
                    initialize=initialize,
                ),
            )
        if pool:
            layers = [
                nn.PointwiseMaxPool(
                    nn.FieldType(self.r2_act, [self.repr] * self.in_planes),
                    kernel_size=k,
                    padding=0,
                    ceil_mode=True,
                )
            ]
        else:
            layers = []

        layers.append(
            block(
                self.in_planes,
                out_planes,
                stride=1,
                down_sample=down_sample,
                N=N,
                initialize=initialize,
            )
        )

        self.in_planes = out_planes

        for i in range(1, block_num):
            layers.append(
                block(
                    self.in_planes,
                    out_planes,
                    stride=1,
                    down_sample=None,
                    N=N,
                    initialize=initialize,
                )
            )

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(
                x,
                nn.FieldType(
                    self.r2_act, self.obs_channel * [self.r2_act.trivial_repr]
                ),
            )

        #  76x76
        x = self.conv1(x)
        x = self.relu(x)
        # 72x72
        x = self.maxpool(x)
        # 36x36
        x = self.layer1(x)
        # 36x36
        x = self.layer2(x)
        # 18x18
        x = self.layer3(x)
        # 9x9
        x = self.layer4(x)
        # 3x3
        out = self.relu_out(self.out_layer(x))
        # 1x1
        return out


class EquiBasicBlock(torch.nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        N=8,
        initialize=True,
        stride=1,
        down_sample=None,
        dilation=1,
        kernel_size=3,
    ):
        super(EquiBasicBlock, self).__init__()
        r2_act = gspaces.rot2dOnR2(N=N)

        rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, [rep] * in_planes)
        feat_type_hid = nn.FieldType(r2_act, [rep] * out_planes)

        self.conv1 = nn.R2Conv(
            feat_type_in,
            feat_type_hid,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) // 2,
            initialize=initialize,
        )

        self.conv2 = nn.R2Conv(
            feat_type_hid,
            feat_type_hid,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) // 2,
            initialize=initialize,
        )

        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.down_sample is not None:
            identity = self.down_sample(identity)

        out += identity
        out = self.relu(out)

        return out


class EquiResNet_18(torch.nn.Module):
    def __init__(
        self, block, layers, obs_channel: int = 3, N=8, n_out: int = 64, initialize=True
    ):
        super(EquiResNet_18, self).__init__()
        self.obs_channel = obs_channel
        self.N = N
        self.r2_act = gspaces.rot2dOnR2(N=N)
        self.repr = self.r2_act.regular_repr
        self.in_planes = n_out // N

        feat_type_in = nn.FieldType(
            self.r2_act, [self.r2_act.trivial_repr] * obs_channel
        )
        feat_type_hid = nn.FieldType(self.r2_act, [self.repr] * self.in_planes)

        self.conv1 = nn.R2Conv(
            feat_type_in,
            feat_type_hid,
            kernel_size=5,
            stride=1,
            padding=0,
            initialize=initialize,
        )

        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.maxpool = nn.PointwiseMaxPool(
            feat_type_hid, kernel_size=2, stride=2, padding=0, ceil_mode=True
        )

        self.layer1 = self._make_layer(
            block, n_out // 8, layers[0], stride=1, N=N, initialize=initialize
        )
        self.layer2 = self._make_layer(
            block,
            n_out // 4,
            layers[1],
            stride=1,
            N=N,
            initialize=initialize,
            pool=True,
        )
        self.layer3 = self._make_layer(
            block,
            n_out // 2,
            layers[2],
            stride=1,
            N=N,
            initialize=initialize,
            pool=True,
        )
        self.layer4 = self._make_layer(
            block, n_out, layers[3], stride=1, N=N, initialize=initialize
        )

        self.out_layer = nn.R2Conv(
            nn.FieldType(self.r2_act, n_out * [self.r2_act.regular_repr]),
            nn.FieldType(self.r2_act, 8 * n_out * [self.r2_act.trivial_repr]),
            kernel_size=3,
            padding=0,
            initialize=initialize,
        )

    def _make_layer(
        self, block, out_planes, block_num, stride=1, N=4, initialize=True, pool=False
    ):
        down_sample = None
        if self.in_planes != out_planes:
            down_sample = nn.SequentialModule(
                nn.R2Conv(
                    nn.FieldType(self.r2_act, [self.repr] * self.in_planes),
                    nn.FieldType(self.r2_act, [self.repr] * out_planes),
                    kernel_size=1,
                    stride=stride,
                    initialize=initialize,
                ),
            )
        if pool:
            layers = [
                nn.PointwiseMaxPool(
                    nn.FieldType(self.r2_act, [self.repr] * self.in_planes),
                    kernel_size=2,
                    padding=0,
                    ceil_mode=True,
                )
            ]
        else:
            layers = []

        layers.append(
            block(
                self.in_planes,
                out_planes,
                stride=1,
                down_sample=down_sample,
                N=N,
                initialize=initialize,
            )
        )

        self.in_planes = out_planes

        for i in range(1, block_num):
            layers.append(
                block(
                    self.in_planes,
                    out_planes,
                    stride=1,
                    down_sample=None,
                    N=N,
                    initialize=initialize,
                )
            )

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(
                x,
                nn.FieldType(
                    self.r2_act, self.obs_channel * [self.r2_act.trivial_repr]
                ),
            )
        # 76x76
        x = self.conv1(x)
        x = self.relu(x)
        # 72x72
        x = self.maxpool(x)
        # 36x36
        x = self.layer1(x)
        x = self.layer2(x)
        # 18x18
        x = self.layer3(x)
        # 9x9
        x = self.layer4(x)
        # 7x7
        out = self.out_layer(x)

        return out


def EquiResnet18(obs_channel=3, N=8, out_fdim=1024, initialize=True):
    out_fdim //= N
    model = EquiResNet_18(
        EquiBasicBlock,
        [2, 2, 2, 2],
        obs_channel=obs_channel,
        N=N,
        initialize=initialize,
        n_out=out_fdim,
    )

    return model


def EquiResnet34(obs_channel=3, N=8, out_fdim=1024, initialize=True):
    out_fdim //= N
    model = EquiResNet_18(
        EquiBasicBlock,
        [3, 4, 6, 3],
        obs_channel=obs_channel,
        N=N,
        initialize=initialize,
        n_out=out_fdim,
    )

    return model


def EquiResnet50(obs_channel=3, N=8, out_fdim=1024, initialize=True):
    out_fdim //= N
    model = EquiResNet_50(
        EquiBottleNeck,
        [3, 4, 6, 3],
        obs_channel=obs_channel,
        N=N,
        initialize=initialize,
        n_out=out_fdim,
    )

    return model


def EquiResnet1x1(obs_channel=3, N=8, out_fdim=1024, initialize=True):
    out_fdim //= N
    model = EquiResNet_50_1x1(
        EquiBottleNeck,
        [3, 4, 6, 3],
        obs_channel=obs_channel,
        N=N,
        initialize=initialize,
        n_out=out_fdim,
    )

    return model


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    x = torch.randn(1, 3, 76, 76).to("cuda")
    x90 = torch.rot90(x, k=1, dims=(2, 3))

    model = EquiResnet50(N=8, obs_channel=x.shape[1], out_fdim=512).to("cuda")
    out = model(x)
    out90 = model(x90)

    out90_0 = torch.rot90(out90.tensor, k=3, dims=(2, 3))

    # number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    # current_memory_allocated
    current_memory = torch.cuda.memory_allocated()
    print(f"Current memory allocated: {current_memory / 1024 ** 2:.2f} MB")

    # max_memory_allocated
    max_memory = torch.cuda.max_memory_allocated()
    print(f"Max memory allocated: {max_memory / 1024 ** 2:.2f} MB")

    print("params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
