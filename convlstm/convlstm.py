"""三层 ConvLSTM：把 notebook In [2] 的单层网络直接扩成三层。

原代码只有::

    self.convLSTM_cell = CLSTM_cell(input_dim, hidden_dim, kernel_size)

这里改成 cell1 / cell2 / cell3。当前层的 ``h_t`` 当作后一层的 ``x_t``。
构造函数签名与原来一致：``ConvLSTM(input_dim, hidden_dim, kernel_size)``。

输入布局::

    inputs: (seq_len, batch, channels, height, width)
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


HiddenState = Tuple[torch.Tensor, torch.Tensor]


def _to_pair(kernel_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    if len(kernel_size) != 2:
        raise ValueError("kernel_size must be an int or a pair (kH, kW)")
    return int(kernel_size[0]), int(kernel_size[1])


class CLSTM_cell(nn.Module):
    """单步 ConvLSTM 单元。调用方式和 notebook 一致::

        hy, hidden_state = cell(inputs[t, ...], hidden_state)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(CLSTM_cell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = _to_pair(kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(self, input_tensor, hidden_state):
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

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # device=self.conv.weight.device: 选择设备
        device = self.conv.weight.device
        dtype = self.conv.weight.dtype
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device, dtype=dtype),
        )


class ConvLSTM(nn.Module):
    """三层 ConvLSTM，由单层 In [2] 改写。

    参数（与原来相同，只是内部变成三层）：

    - input_dim: 输入特征维数
    - hidden_dim: 隐藏层状态的维数（int，三层共用；或长度为 3 的列表）
    - kernel_size / filter_size: 卷积核尺寸
    """

    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTM, self).__init__()
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim, hidden_dim]
        else:
            hidden_dim = [int(v) for v in hidden_dim]
            if len(hidden_dim) != 3:
                raise ValueError("三层网络的 hidden_dim 必须是 int，或长度为 3 的列表")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 3
        kernel_size = _to_pair(kernel_size)

        # 单层时是 self.convLSTM_cell = CLSTM_cell(input_dim, hidden_dim, kernel_size)
        # 三层：第 1 层读 x_t，第 2、3 层读上一层的 h_t
        self.cell1 = CLSTM_cell(input_dim, hidden_dim[0], kernel_size)
        self.cell2 = CLSTM_cell(hidden_dim[0], hidden_dim[1], kernel_size)
        self.cell3 = CLSTM_cell(hidden_dim[1], hidden_dim[2], kernel_size)
        self.cell_list = nn.ModuleList([self.cell1, self.cell2, self.cell3])

    def _init_hidden(self, batch_size, image_size):
        # 初始化返回的隐状态与后面使用的隐状态组成成分一样；
        # 当前层的 h_t 会当作后一层的 x_t，所以三层各自一份 (h, c)
        return [
            self.cell1.init_hidden(batch_size, image_size),
            self.cell2.init_hidden(batch_size, image_size),
            self.cell3.init_hidden(batch_size, image_size),
        ]

    def _run_layer(self, cell, inputs, hidden_state):
        # 取出时间长度，供下面按时间步更新 (h, c)
        s = inputs.size(0)
        output_inner = []
        for t in range(s):
            # 每一个时间步都更新 hidden_state 中的 h, c
            # 当前时间步的输出 hidden_state，又作为下一时间步的输入 hidden_state
            hy, hidden_state = cell(inputs[t, ...], hidden_state)
            output_inner.append(hy)
        return torch.stack(output_inner), hidden_state

    def forward(self, inputs=None, hidden_state=None):
        if inputs is None:
            raise ValueError("inputs is required")

        # 取出图片的数据，供下面初始化使用
        s, b, _, h, w = inputs.size()
        if hidden_state is None:
            # 初始化 hidden_state，利用后面和 lstm 单元中的初始化函数
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        if len(hidden_state) != 3:
            raise ValueError("三层网络的 hidden_state 必须包含 3 份 (h, c)")

        # 当前层的 h_t 会当作后一层的 x_t 作为输入使用
        y1, s1 = self._run_layer(self.cell1, inputs, hidden_state[0])
        y2, s2 = self._run_layer(self.cell2, y1, hidden_state[1])
        y3, s3 = self._run_layer(self.cell3, y2, hidden_state[2])
        return y3, [s1, s2, s3]


if __name__ == "__main__":
    seq_len, batch, channels, height, width = 10, 2, 1, 32, 32
    model = ConvLSTM(input_dim=channels, hidden_dim=16, kernel_size=3)
    x = torch.randn(seq_len, batch, channels, height, width)
    y, states = model(x)
    print("output:", tuple(y.shape))
    print("num layers:", len(states))
    print("last hidden:", tuple(states[-1][0].shape))
