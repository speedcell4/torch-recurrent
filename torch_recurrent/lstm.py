from typing import Tuple

import torch
from torch import nn

HX = Tuple[torch.Tensor, torch.Tensor]


# TODO batch_first
# TODO PackedSequence
class LSTM(nn.LSTM):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = False,
                 batch_first: bool = True, dropout: float = 0, bidirectional: bool = False) -> None:
        super(LSTM, self).__init__(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        self.h0 = nn.Parameter(torch.Tensor(num_layers * num_directions, 1, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(num_layers * num_directions, 1, hidden_size))

    def hx(self, batch_size: int) -> HX:
        h0 = self.h0.expand(-1, batch_size, -1)
        c0 = self.c0.expand(-1, batch_size, -1)
        return h0, c0

    def forward(self, input: torch.Tensor, hx: HX = None) -> Tuple[torch.Tensor, HX]:
        if hx is None:
            hx = self.hx(input.size(0))
        return super(LSTM, self).forward(input, hx)


