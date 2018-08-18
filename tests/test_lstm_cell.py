import torch
from hypothesis import given, strategies as st

from torch_recurrent.lstm_cell import LSTMCell

hyper = dict(
    batch=st.integers(1, 20),
    input_size=st.integers(20, 50),
    hidden_size=st.integers(20, 50),
    bias=st.booleans(),
)


@given(**hyper)
def test_lstm_cell(batch, input_size, hidden_size, bias):
    cell = LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    inputs = torch.rand(batch, input_size)
    ht, ct = cell(inputs)
    assert ht.size() == (batch, hidden_size)
    assert ct.size() == (batch, hidden_size)


@given(**hyper)
def test_lstm_cell_with_hx(batch, input_size, hidden_size, bias):
    cell = LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias)
    inputs = torch.rand(batch, input_size)
    h0 = torch.rand(batch, hidden_size)
    c0 = torch.rand(batch, hidden_size)
    ht, ct = cell(inputs, (h0, c0))
    assert ht.size() == (batch, hidden_size)
    assert ct.size() == (batch, hidden_size)
