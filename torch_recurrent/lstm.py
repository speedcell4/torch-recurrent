from typing import Tuple, Union

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from torch_recurrent import keras_lstm_

IO = Union[torch.Tensor, PackedSequence]
HX = Tuple[torch.Tensor, torch.Tensor]


class LSTM(nn.LSTM):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = False,
                 batch_first: bool = True, dropout: float = 0, bidirectional: bool = False) -> None:
        super(LSTM, self).__init__(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional,
        )

        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_size * self.num_directions
        self.h0 = nn.Parameter(torch.Tensor(num_layers * self.num_directions, 1, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(num_layers * self.num_directions, 1, hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        keras_lstm_(self)

    def hx(self, batch_size: int) -> HX:
        h0 = self.h0.expand(-1, batch_size, -1).contiguous()
        c0 = self.c0.expand(-1, batch_size, -1).contiguous()
        return h0, c0

    def forward(self, input: IO, hx: HX = None) -> Tuple[IO, HX]:
        if hx is None:
            if isinstance(input, PackedSequence):
                batch_size = input.batch_sizes[0].item()
            elif self.batch_first:
                batch_size = input.size(0)
            else:
                batch_size = input.size(1)
            hx = self.hx(batch_size)
        return super(LSTM, self).forward(input, hx)

    def reduce(self, input: IO) -> torch.Tensor:
        _, (output, _) = self.__call__(input, None)
        output = output.view(self.num_layers, self.num_directions, -1, self.hidden_size)

        if self.num_directions == 1:
            return output[-1, 0, :, :]
        return torch.cat([output[-1, 0, :, :], output[-1, 1, :, :]], dim=-1)

    def transduce(self, input: IO) -> IO:
        output, _ = self.__call__(input, None)
        return output
