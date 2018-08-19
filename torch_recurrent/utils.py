from typing import List, Tuple, Union

import torch
from torch.nn import init
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence


def keras_lstm_(module: Union[nn.LSTM, nn.LSTMCell]) -> None:
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
                init.xavier_uniform_(param)
            if name == 'c0':
                init.xavier_uniform_(param)


def unsorted_pack_sequence(sequences: List, dtype=torch.long,
                           device=torch.device('cpu')) -> Tuple[PackedSequence, torch.Tensor]:
    sequences = [
        seq if torch.is_tensor(seq) else torch.tensor(seq, dtype=dtype, device=device)
        for seq in sequences
    ]
    idx = list(range(len(sequences)))
    idx = sorted(idx, key=lambda item: sequences[item].size(0), reverse=True)
    inv = [None] * len(sequences)
    for x, y in enumerate(idx):  # type:int, int
        inv[y] = x
    sequences = pack_sequence([sequences[ix] for ix in idx])
    return sequences, torch.tensor(inv, dtype=torch.long, device=device)
