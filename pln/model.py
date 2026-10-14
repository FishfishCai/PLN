from typing import Dict, Sequence

import torch
import torch.nn as nn

from utils.base_model import BaseModel


class PLNModel(BaseModel):
    """PolarLiteNet, two fully connected blocks and a 1D conv block coupled by a residual skip that reconstruct ``S0`` and ``S1'`` jointly."""

    def __init__(
        self,
        y_dim: int,
        x_dim: int,
        rec_hidden: Sequence[int] = (1024, 512),
        conv_channels: Sequence[int] = (8, 32, 64),
        fc_hidden: Sequence[int] = (1024, 512),
        use_cnn: bool = True,
        use_skip: bool = True,
        separate_heads: bool = False,
        dropout: float = 0.25,
    ) -> None:
        """
        Build the coarse-inversion MLP and, optionally, the convolutional refinement branch.

        Every architecture choice is a constructor argument, so the capacity and component
        ablations are driven entirely by the modeling config.

        Parameters
        ----------
        y_dim : int
            Length of the normalized measurement vector, ``2 * n_devices``.
        x_dim : int
            Length of the joint output ``[S0, S1']``, ``2 * n_wavelengths``.
        rec_hidden : Sequence[int]
            Hidden widths of the coarse-inversion MLP. Default is ``(1024, 512)``.
        conv_channels : Sequence[int]
            Output channels of the three 1D conv layers. Default is ``(8, 32, 64)``.
        fc_hidden : Sequence[int]
            Hidden widths of the refinement head or heads after the conv block.
            Default is ``(1024, 512)``.
        use_cnn : bool
            Whether to add the conv refinement branch on top of the coarse estimate.
            Default is True.
        use_skip : bool
            Whether to add the coarse estimate back to the refinement, used only when
            ``use_cnn`` is True. Default is True.
        separate_heads : bool
            Whether to refine ``S0`` and ``S1'`` with two independent heads instead of one
            joint head, used only when ``use_cnn`` is True. Default is False.
        dropout : float
            Dropout probability after every hidden activation of the fully connected
            blocks, active only in training. Default is 0.25.
        """
        super().__init__()
        self.use_cnn = use_cnn
        self.use_skip = use_skip
        self.separate_heads = separate_heads
        self.n_wavelengths = x_dim // 2

        self.rec_fc = self._make_mlp(y_dim, rec_hidden, x_dim, dropout)

        if self.use_cnn:
            c1, c2, c3 = conv_channels
            self.conv = nn.Sequential(
                nn.Conv1d(1, c1, kernel_size=25, padding=12),
                nn.MaxPool1d(kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(c1, c2, kernel_size=11, padding=5),
                nn.MaxPool1d(kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(c2, c3, kernel_size=5, padding=2),
                nn.MaxPool1d(kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
            )

            with torch.no_grad():
                conv_out_dim = self.conv(torch.zeros(1, 1, x_dim)).shape[1]

            if self.separate_heads:
                self.fc_s0 = self._make_mlp(conv_out_dim, fc_hidden, self.n_wavelengths, dropout)
                self.fc_s1 = self._make_mlp(conv_out_dim, fc_hidden, self.n_wavelengths, dropout)
            else:
                self.fc = self._make_mlp(conv_out_dim, fc_hidden, x_dim, dropout)

    @staticmethod
    def _make_mlp(
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
        dropout: float = 0.25,
    ) -> nn.Sequential:
        """
        Stack ``Linear -> ReLU -> Dropout`` blocks followed by a final linear layer.

        Parameters
        ----------
        in_dim : int
            Input feature length.
        hidden : Sequence[int]
            Width of each hidden layer, in order.
        out_dim : int
            Output feature length.
        dropout : float
            Dropout probability after every hidden activation. Default is 0.25.

        Returns
        -------
        mlp : nn.Sequential
            The assembled multilayer perceptron.
        """
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(p=dropout)]
        layers.append(nn.Linear(dims[-1], out_dim))
        return nn.Sequential(*layers)

    def forward(
        self,
        y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct the joint ``[S0, S1']`` vector from normalized measurements.

        Parameters
        ----------
        y : torch.Tensor
            Normalized measurements ``[I_x, I_y]`` of shape ``(batch, y_dim)``.

        Returns
        -------
        outputs : Dict[str, torch.Tensor]
            ``"x_pred"`` of shape ``(batch, x_dim)`` in ``[-1, 1]``, whose first half is
            ``S0`` and second half ``S1'``.
        """
        xt = self.rec_fc(y)
        if self.use_cnn:
            feat = self.conv(xt[:, None, :])
            if self.separate_heads:
                fx = torch.cat([self.fc_s0(feat), self.fc_s1(feat)], dim=1)
            else:
                fx = self.fc(feat)
            z = fx + xt if self.use_skip else fx
        else:
            z = xt
        # 2 * sigmoid(2z) - 1 equals tanh(z)
        x_pred = 2.0 * torch.sigmoid(2.0 * z) - 1.0
        return {"x_pred": x_pred}
