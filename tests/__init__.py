from hypothesis import strategies as st

BATCH = st.integers(1, 20)
SEQ_LEN = st.integers(1, 50)

NUM_LAYERS = st.integers(1, 3)

INPUT_SIZE = st.integers(20, 50)
HIDDEN_SIZE = st.integers(20, 50)

BIAS = st.booleans()
DROPOUT = st.floats(0., 1.)

BIDIRECTIONAL = st.booleans()

SEQ_LENS = st.lists(SEQ_LEN, min_size=1, max_size=20)
SORTED_SEQ_LENS = SEQ_LENS.map(sorted).map(reversed).map(list)
