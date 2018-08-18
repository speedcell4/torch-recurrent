import torch
from hypothesis import given, strategies as st

from torch_recurrent import LSTM

hyper = dict(
    seqlen=st.integers(20, 50),
    batch=st.integers(1, 20),
    input_size=st.integers(20, 50),
    hidden_size=st.integers(20, 50),
    bias=st.booleans(),
    num_layers=st.integers(1, 3),
    dropout=st.floats(0., 1.),
    bidirectional=st.booleans(),
)


@given(**hyper)
def test_lstm(seqlen, batch, input_size, hidden_size, bias, num_layers, dropout, bidirectional):
    if num_layers == 1:
        dropout = 0
    num_directions = 2 if bidirectional else 1

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=True, dropout=dropout, bidirectional=bidirectional)

    inputs = torch.rand(batch, seqlen, input_size)
    outputs, (h_n, c_n) = rnn(inputs)

    assert outputs.size() == (batch, seqlen, hidden_size * num_directions)
    assert h_n.size() == (num_layers * num_directions, batch, hidden_size)
    assert c_n.size() == (num_layers * num_directions, batch, hidden_size)


@given(**hyper)
def test_lstm_with_hx(seqlen, batch, input_size, hidden_size, bias, num_layers, dropout, bidirectional):
    if num_layers == 1:
        dropout = 0
    num_directions = 2 if bidirectional else 1

    rnn = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
               batch_first=True, dropout=dropout, bidirectional=bidirectional)

    inputs = torch.rand(batch, seqlen, input_size)
    h_0 = torch.rand(num_layers * num_directions, batch, hidden_size)
    c_0 = torch.rand(num_layers * num_directions, batch, hidden_size)
    outputs, (h_n, c_n) = rnn(inputs, (h_0, c_0))

    assert outputs.size() == (batch, seqlen, hidden_size * num_directions)
    assert h_n.size() == (num_layers * num_directions, batch, hidden_size)
    assert c_n.size() == (num_layers * num_directions, batch, hidden_size)
