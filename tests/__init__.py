from hypothesis import strategies as st

max_batch = 20
max_len = 50

BATCH = st.integers(1, max_batch)
NUM_LAYERS = st.integers(1, 3)

INPUT_SIZE = st.integers(20, 50)
HIDDEN_SIZE = st.integers(20, 50)

BIAS = st.booleans()
DROPOUT = st.floats(0., 1.)

BIDIRECTIONAL = st.booleans()

SEQ_LEN = st.integers(1, max_len)
SEQ_LENS = st.lists(SEQ_LEN, min_size=1, max_size=max_batch)
SORTED_SEQ_LENS = SEQ_LENS.map(sorted).map(reversed).map(list)

VOCAB = st.integers(0, 1200)
SEQUENCE = st.lists(VOCAB, min_size=1, max_size=max_len)
SEQUENCES = st.lists(SEQUENCE, min_size=1, max_size=max_batch)
