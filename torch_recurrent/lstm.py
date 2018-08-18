from typing import Tuple

import torch
from torch.nn import init
from torch import nn


class LSTMCell(nn.LSTMCell):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(LSTMCell, self).__init__(
            input_size=input_size, hidden_size=hidden_size, bias=bias,
        )
        self.h0 = nn.Parameter(torch.Tensor(1, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = (6.0 / self.hidden_size) ** 0.5
        with torch.no_grad():
            init.xavier_uniform_(self.weight_ih)
            init.orthogonal_(self.weight_hh)
            if getattr(self, 'bias_ih', None) is not None:
                init.constant_(self.bias_ih, 0.)
                self.bias_ih[self.hidden_size:self.hidden_size * 2] = 0.5
            if getattr(self, 'bias_hh', None) is not None:
                init.constant_(self.bias_hh, 0.)
                self.bias_hh[self.hidden_size:self.hidden_size * 2] = 0.5
            if getattr(self, 'h0', None) is not None:
                init.uniform_(self.h0, -std, +std)
            if getattr(self, 'c0', None) is not None:
                init.uniform_(self.c0, -std, +std)

    def forward(self, input, hx=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            h0 = self.h0.expand(input.size(0), -1)
            c0 = self.c0.expand(input.size(0), -1)
            hx = (h0, c0)
        return super(LSTMCell, self).forward(input=input, hx=hx)
