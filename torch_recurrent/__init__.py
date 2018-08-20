import os

import torch

from .utils import keras_lstm_, unsorted_pack_sequence
from .lstm_cell import LSTMCell
from .lstm import LSTM

if torch.cuda.is_available():
    if 'PYTEST_DEVICE' in os.environ:
        device = os.environ['PYTEST_DEVICE']
        torch.cuda.set_device(int(device))
        device = torch.device(f'cuda:{device}')
else:
    device = torch.device('cpu')
