from hypothesis import given
import torch
from torch.nn.utils.rnn import pack_sequence

from tests import *
from torch_recurrent import LSTM, device


@given(
    seq_len=SEQ_LEN,
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
    batch_first=BATCH_FIRST,
)
def test_lstm_with_hx(seq_len, batch, input_size, hidden_size, bias, num_layers, batch_first, dropout, bidirectional):
    if num_layers == 1:
        dropout = 0
    num_directions = 2 if bidirectional else 1

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    if batch_first:
        inputs = torch.rand(batch, seq_len, input_size)
    else:
        inputs = torch.rand(seq_len, batch, input_size)
    h_0 = torch.rand(num_layers * num_directions, batch, hidden_size)
    c_0 = torch.rand(num_layers * num_directions, batch, hidden_size)

    rnn = rnn.to(device)
    inputs = inputs.to(device)
    h_0 = h_0.to(device)
    c_0 = c_0.to(device)

    outputs, (h_n, c_n) = rnn(inputs, (h_0, c_0))

    if batch_first:
        assert outputs.size() == (batch, seq_len, rnn.output_dim)
    else:
        assert outputs.size() == (seq_len, batch, rnn.output_dim)
    assert h_n.size() == (num_layers * num_directions, batch, hidden_size)
    assert c_n.size() == (num_layers * num_directions, batch, hidden_size)


@given(
    seq_len=SEQ_LEN,
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
    batch_first=BATCH_FIRST,
)
def test_lstm(seq_len, batch, input_size, hidden_size, bias, num_layers, batch_first, dropout, bidirectional):
    if num_layers == 1:
        dropout = 0
    num_directions = 2 if bidirectional else 1

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    if batch_first:
        inputs = torch.rand(batch, seq_len, input_size)
    else:
        inputs = torch.rand(seq_len, batch, input_size)

    rnn = rnn.to(device)
    inputs = inputs.to(device)

    outputs, (h_n, c_n) = rnn(inputs)

    if batch_first:
        assert outputs.size() == (batch, seq_len, rnn.output_dim)
    else:
        assert outputs.size() == (seq_len, batch, rnn.output_dim)
    assert h_n.size() == (num_layers * num_directions, batch, hidden_size)
    assert c_n.size() == (num_layers * num_directions, batch, hidden_size)


@given(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
    seq_lens=SORTED_SEQ_LENS,
)
def test_lstm_reduce_with_pack(
        input_size, hidden_size, bias, num_layers, dropout, bidirectional, seq_lens):
    if num_layers == 1:
        dropout = 0
    batch = len(seq_lens)

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=True, dropout=dropout, bidirectional=bidirectional)
    inputs = pack_sequence([torch.rand(seq_len, input_size) for seq_len in seq_lens])

    rnn = rnn.to(device)
    inputs = inputs.to(device)

    outputs = rnn.reduce(inputs)

    assert outputs.size() == (batch, rnn.output_dim)


@given(
    seq_len=SEQ_LEN,
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
)
def test_lstm_reduce(seq_len, batch, input_size, hidden_size, bias, num_layers, dropout, bidirectional):
    if num_layers == 1:
        dropout = 0

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=True, dropout=dropout, bidirectional=bidirectional)

    inputs = torch.rand(batch, seq_len, input_size)
    rnn = rnn.to(device)
    inputs = inputs.to(device)

    outputs = rnn.reduce(inputs)

    assert outputs.size() == (batch, rnn.output_dim)


@given(
    seq_len=SEQ_LEN,
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
)
def test_lstm_transduce_with_pack(seq_len, batch, input_size, hidden_size, bias, num_layers, dropout, bidirectional):
    if num_layers == 1:
        dropout = 0

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=True, dropout=dropout, bidirectional=bidirectional)

    inputs = torch.rand(batch, seq_len, input_size)
    rnn = rnn.to(device)
    inputs = inputs.to(device)

    outputs = rnn.transduce(inputs)

    assert outputs.size() == (batch, seq_len, rnn.output_dim)


@given(
    seq_len=SEQ_LEN,
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
)
def test_lstm_transduce(seq_len, batch, input_size, hidden_size, bias, num_layers, dropout, bidirectional):
    if num_layers == 1:
        dropout = 0

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=True, dropout=dropout, bidirectional=bidirectional)

    inputs = torch.rand(batch, seq_len, input_size)
    rnn = rnn.to(device)
    inputs = inputs.to(device)

    outputs = rnn.transduce(inputs)

    assert outputs.size() == (batch, seq_len, rnn.output_dim)
