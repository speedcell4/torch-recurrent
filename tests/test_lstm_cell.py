import torch
from hypothesis import given

from torch_recurrent.lstm_cell import LSTMCell
from tests import *


@given(
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
)
def test_lstm_cell(batch, input_size, hidden_size, bias):
    cell = LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    inputs = torch.rand(batch, input_size)
    ht, ct = cell(inputs)
    assert ht.size() == (batch, hidden_size)
    assert ct.size() == (batch, hidden_size)


@given(
    batch=BATCH,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    bias=BIAS,
)
def test_lstm_cell_with_hx(batch, input_size, hidden_size, bias):
    cell = LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    inputs = torch.rand(batch, input_size)
    h0 = torch.rand(batch, hidden_size)
    c0 = torch.rand(batch, hidden_size)
    ht, ct = cell(inputs, (h0, c0))
    assert ht.size() == (batch, hidden_size)
    assert ct.size() == (batch, hidden_size)
