import torch
from hypothesis import given

from torch_recurrent import LSTMCell, device
from tests import *


@given(
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
)
def test_lstm_cell(batch, input_size, hidden_size, bias):
    rnn = LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    inputs = torch.rand(batch, input_size)

    rnn = rnn.to(device)
    inputs = inputs.to(device)

    ht, ct = rnn(inputs)
    assert ht.size() == (batch, hidden_size)
    assert ct.size() == (batch, hidden_size)


@given(
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
)
def test_lstm_cell_with_hx(batch, input_size, hidden_size, bias):
    rnn = LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    inputs = torch.rand(batch, input_size)
    h_0 = torch.rand(batch, hidden_size)
    c_0 = torch.rand(batch, hidden_size)

    rnn = rnn.to(device)
    inputs = inputs.to(device)
    h_0 = h_0.to(device)
    c_0 = c_0.to(device)

    ht, ct = rnn(inputs, (h_0, c_0))
    assert ht.size() == (batch, hidden_size)
    assert ct.size() == (batch, hidden_size)
