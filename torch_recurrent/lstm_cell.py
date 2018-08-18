from typing import Callable, Tuple

import torch
from torch import nn

from torch_recurrent import keras_lstm_

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
        self.output_dim = hidden_size
        self.h0 = nn.Parameter(torch.Tensor(1, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        return keras_lstm_(self)

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
