from typing import List, Union

import torch
from torch.nn import init
from torch import nn
from torch.nn.utils.rnn import pack_sequence


def keras_lstm_(module: Union[nn.LSTM, nn.LSTMCell]) -> None:
    std = (6.0 / module.hidden_size) ** 0.5

    for name, param in module.named_parameters():  # type: str, nn.Parameter
        with torch.no_grad():
            if name.startswith('weight_hh'):
                init.orthogonal_(param)
            if name.startswith('weight_ih'):
                init.xavier_uniform_(param)
            if name.startswith('bias_'):
                init.constant_(param, 0.)
                param[module.hidden_size:module.hidden_size * 2] = 0.5
            if name == 'h0':
                init.uniform_(param, -std, +std)
            if name == 'c0':
                init.uniform_(param, -std, +std)


def unsorted_pack_sequence(sequences: List, dtype=torch.long, device=torch.device('cpu')):
    sequences = [
        seq if torch.is_tensor(seq) else torch.tensor(seq, dtype=dtype, device=device)
        for seq in sequences
    ]
    idx = list(range(len(sequences)))
    idx = sorted(idx, key=lambda item: sequences[item].size(0), reverse=True)
    inv = [0] * len(sequences)
    for x, y in enumerate(idx):
        inv[y] = x
    return pack_sequence([sequences[ix] for ix in idx]), torch.tensor(inv, dtype=dtype, device=device)


