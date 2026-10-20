"""Stacked ConvLSTM (default: 3 layers).

This is a multilayer extension of the single-layer notebook cell:

    class ConvLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, kernel_size):
            ...
            self.convLSTM_cell = CLSTM_cell(input_dim, hidden_dim, kernel_size)

The single-layer network walks time with one cell. The three-layer network
keeps that same cell and time loop, then stacks three cells so that the current
layer's hidden state ``h_t`` is used as the next layer's input ``x_t``.

Tensor layout matches the original notebook::

    inputs: (seq_len, batch, channels, height, width)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


KernelSize = Union[int, Tuple[int, int], List[Tuple[int, int]]]
HiddenDim = Union[int, Sequence[int]]
HiddenState = Tuple[torch.Tensor, torch.Tensor]


def _to_pair(kernel_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    if len(kernel_size) != 2:
        raise ValueError("kernel_size must be an int or a pair (kH, kW)")
    return int(kernel_size[0]), int(kernel_size[1])


class CLSTM_cell(nn.Module):
    """Single-timestep ConvLSTM cell used by the stacked ConvLSTM.

    Calling convention matches the notebook::

        hy, hidden_state = cell(inputs[t, ...], hidden_state)

    where ``hidden_state`` is the ``(h, c)`` pair.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Union[int, Tuple[int, int]]):
        super(CLSTM_cell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = _to_pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Concatenate x_t and h_{t-1} along channels, then emit the 4 LSTM gates.
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: HiddenState,
    ) -> Tuple[torch.Tensor, HiddenState]:
        h_cur, c_cur = hidden_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> HiddenState:
        height, width = image_size
        device = self.conv.weight.device
        dtype = self.conv.weight.dtype
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
        )


class ConvLSTM(nn.Module):
    """Three-layer (by default) ConvLSTM built from stacked ``CLSTM_cell`` units.

    Parameters
    ----------
    input_dim:
        Number of input feature channels.
    hidden_dim:
        Hidden-state channels. An int is broadcast to every layer; a sequence
        must have length ``num_layers``.
    kernel_size:
        Convolution kernel size. An int or ``(kH, kW)`` is broadcast; a list
        must have one pair per layer.
    num_layers:
        Network depth. Defaults to 3.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: HiddenDim,
        kernel_size: KernelSize,
        num_layers: int = 3,
    ):
        super(ConvLSTM, self).__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.kernel_size = self._extend_kernel_size(kernel_size, num_layers)

        cell_list = []
        for i in range(self.num_layers):
            # Layer 0 reads the raw input; later layers read the previous h_t.
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                CLSTM_cell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def _init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> List[HiddenState]:
        # One (h, c) pair per layer, on the same device as the corresponding cell.
        return [
            self.cell_list[i].init_hidden(batch_size, image_size)
            for i in range(self.num_layers)
        ]

    def forward(
        self,
        inputs: Optional[torch.Tensor] = None,
        hidden_state: Optional[List[HiddenState]] = None,
    ) -> Tuple[torch.Tensor, List[HiddenState]]:
        if inputs is None:
            raise ValueError("inputs is required")

        # Unpack image data for hidden-state initialization.
        # inputs: (seq_len, batch, channels, height, width)
        s, b, _, h, w = inputs.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        if len(hidden_state) != self.num_layers:
            raise ValueError(
                f"hidden_state must have {self.num_layers} layer states, got {len(hidden_state)}"
            )

        cur_layer_input = inputs
        last_state_list: List[HiddenState] = []

        for layer_idx in range(self.num_layers):
            output_inner = []
            layer_state = hidden_state[layer_idx]
            for t in range(s):
                # Each timestep updates (h, c). The current layer's h_t is
                # stacked and used as x_t for the next layer.
                hy, layer_state = self.cell_list[layer_idx](cur_layer_input[t, ...], layer_state)
                output_inner.append(hy)

            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
            last_state_list.append(layer_state)

        return cur_layer_input, last_state_list

    @staticmethod
    def _extend_for_multilayer(param: HiddenDim, num_layers: int) -> List[int]:
        if isinstance(param, int):
            return [param] * num_layers
        values = [int(v) for v in param]
        if len(values) != num_layers:
            raise ValueError(
                f"hidden_dim length ({len(values)}) must equal num_layers ({num_layers})"
            )
        return values

    @staticmethod
    def _extend_kernel_size(kernel_size: KernelSize, num_layers: int) -> List[Tuple[int, int]]:
        if isinstance(kernel_size, int):
            return [_to_pair(kernel_size)] * num_layers
        if isinstance(kernel_size, tuple) and len(kernel_size) == 2 and isinstance(kernel_size[0], int):
            return [_to_pair(kernel_size)] * num_layers
        kernels = [_to_pair(k) for k in kernel_size]  # type: ignore[union-attr]
        if len(kernels) != num_layers:
            raise ValueError(
                f"kernel_size length ({len(kernels)}) must equal num_layers ({num_layers})"
            )
        return kernels


if __name__ == "__main__":
    seq_len, batch, channels, height, width = 10, 2, 1, 32, 32
    model = ConvLSTM(input_dim=channels, hidden_dim=16, kernel_size=3, num_layers=3)
    x = torch.randn(seq_len, batch, channels, height, width)
    y, states = model(x)
    print("output:", tuple(y.shape))
    print("num layers:", len(states))
    print("last hidden:", tuple(states[-1][0].shape))
