from hypothesis import given
from torch.nn.utils.rnn import pad_packed_sequence

from tests import SEQUENCES
from torch_recurrent import unsorted_pack_sequence


@given(
    sequences=SEQUENCES,
)
def test_unsorted_pack_sequence(sequences):
    a, inv = unsorted_pack_sequence(sequences)
    data, lens = pad_packed_sequence(a, batch_first=True)
    for datum, l, seq in zip(data[inv], lens[inv], sequences):
        assert datum[:l].tolist() == seq
