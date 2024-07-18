import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, do_weight_norm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, do_weight_norm=True, max_norm: float = 1, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm

        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


# Support classes for FBNet Implementation
class VarLayer(nn.Module):
    """
    The variance layer: calculates the variance of the data along given 'dim'
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim=self.dim, keepdim=True)


class LogVarLayer(nn.Module):
    """
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


class Swish(nn.Module):
    """
    The swish layer: implements the swish activation function
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    """
    FBNet with seperate variance for every 1s.
    The data input is in a form of batch x 1 x chan x time x filterBand

    Args:
        n_chan: Number of EEG channels
        n_time: Number of time points
        m: Number of spatial filters
        stride_factor: Time window
    """

    def __init__(
        self,
        n_chan: int,
        n_class: int,
        do_weight_norm: bool = True,
        m=32,
        n_bands=9,
        stride_factor=4,
        temporal_layer=LogVarLayer,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.m = m
        self.stride_factor = stride_factor

        # create all the parrallel SCBc
        """
        The spatial convolution block
        m : number of sptatial filters.
        n_bands: number of bands in the data
        """
        self.scb = nn.Sequential(
            Conv2dWithConstraint(
                n_bands, m * n_bands, (n_chan, 1), groups=n_bands, max_norm=2, do_weight_norm=do_weight_norm, padding=0
            ),
            nn.BatchNorm2d(m * n_bands),
            Swish(),
        )

        # Formulate the temporal agreegator
        self.temporal_layer = temporal_layer(dim=3)

        # The final fully connected layer
        self.last_layer = nn.Sequential(
            LinearWithConstraint(
                self.m * self.n_bands * self.stride_factor, n_class, max_norm=0.5, do_weight_norm=do_weight_norm
            ),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.stride_factor, int(x.shape[3] / self.stride_factor)])
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)

        return x
