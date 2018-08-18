from typing import Callable, Tuple

import torch
from torch.nn import init
from torch import nn

HX = Tuple[torch.Tensor, torch.Tensor]


class Statue(object):
    def __init__(self, hx: HX, forward: Callable[[torch.Tensor, HX], HX]) -> None:
        super(Statue, self).__init__()
        self.hx = hx
        self.forward = forward

    @property
    def output(self) -> torch.Tensor:
        return self.hx[0]

    def __call__(self, x: torch.Tensor) -> 'Statue':
        hx = self.forward(x, self.hx)
        return Statue(hx, self.forward)


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

    def hx(self, batch_size: int) -> HX:
        h0 = self.h0.expand(batch_size, -1)
        c0 = self.c0.expand(batch_size, -1)
        return h0, c0

    @property
    def statue(self) -> 'Statue':
        return Statue(self.hx(1), self.__call__)

    def forward(self, input: torch.Tensor, hx: HX = None) -> HX:
        if hx is None:
            hx = self.hx(input.size(0))
        return super(LSTMCell, self).forward(input=input, hx=hx)
