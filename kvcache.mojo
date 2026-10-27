from tenmo.tensor import Tensor
from std.testing import assert_true, assert_raises
from tenmo.shared.indexhelper import i, s, e


struct KVCache[dtype: DType]:
    var keys: Tensor[Self.dtype]  # (B, n_head, capacity, head_dim)
    var values: Tensor[Self.dtype]  # (B, n_head, capacity, head_dim)
    var length: Int  # live positions, 0..capacity
    var capacity: Int  # == seq_length

    def __init__(
        out self, batch_size: Int, n_head: Int, capacity: Int, head_dim: Int
    ):
        self.keys = Tensor[Self.dtype](batch_size, n_head, capacity, head_dim)
        self.values = Tensor[Self.dtype](batch_size, n_head, capacity, head_dim)
        self.capacity = capacity
        self.length = 0

    def reset(mut self):
        # Only the length matters; slots are overwritten on the next append.
        self.length = 0

    def is_full(self) -> Bool:
        # True when every slot is occupied. generate_cached uses this to
        # decide between appending a single new token and rebuilding the
        # window (reset + re-prefill) with correct position embeddings.
        return self.length == self.capacity

    def append(
        mut self, k_new: Tensor[Self.dtype], v_new: Tensor[Self.dtype]
    ) raises:
        """Write T_new new positions at [length : length + T_new].

        Precondition: length + T_new <= capacity. `append` never evicts or
        rebuilds on its own; the caller (generate_cached) checks is_full()
        first and rebuilds when the window is full.
        """
        ref k_new_shape = k_new.shape()
        var T_new = k_new_shape[2]
        ref self_keys_shape = self.keys.shape()

        # Shape checks first: with broadcast-capable fill, a wrong B/n_head/
        # head_dim would silently misalign instead of crashing. These are the
        # correctness mechanism, not defensive noise.
        assert_true(k_new_shape[0] == self_keys_shape[0])  # B
        assert_true(k_new_shape[1] == self_keys_shape[1])  # n_head
        assert_true(k_new_shape[3] == self_keys_shape[3])  # head_dim
        assert_true(v_new.shape() == k_new_shape)  # v mirrors k
        assert_true(self.length + T_new <= self.capacity)  # room to write

        # fill(dest_slices..., src): copy `k_new` into keys[:, :, lo:hi, :].
        self.keys.fill(k_new, e(), s(self.length, self.length + T_new), s())
        self.values.fill(v_new, e(), s(self.length, self.length + T_new), s())
        self.length += T_new

    def get(ref self) raises -> Tuple[Tensor[Self.dtype], Tensor[Self.dtype]]:
        """Return views of the live prefix: (B, n_head, length, head_dim).

        Precondition: length > 0. tenmo cannot represent a zero-length axis
        (Shape(0) is invalid), so an empty cache has no tensor view. Attention
        only reads the cache after prefill has appended at least one position,
        so this precondition is satisfied in the generation flow.
        """
        assert_true(self.length > 0)
        return (
            self.keys[e(), s(0, self.length), s()],
            self.values[e(), s(0, self.length), s()],
        )

    def evict_oldest(mut self, n: Int = 1) raises:
        """Drop the n oldest positions and shift the rest down.

        Used only when the cache is full and a new position arrives (the
        sliding-window step). n must be in [0, length].
        """
        assert_true(n >= 0)
        assert_true(n <= self.length)
        if n == 0:
            return
        var remaining = self.length - n
        # Source [n:length] and destination [0:remaining] overlap when
        # n < length. Clone the source before filling; otherwise the write
        # clobbers slots the read hasn't reached yet.
        var k_src = self.keys[e(), s(n, self.length), s()].clone()
        var v_src = self.values[e(), s(n, self.length), s()].clone()
        self.keys.fill(k_src^, e(), s(0, remaining), s())
        self.values.fill(v_src^, e(), s(0, remaining), s())
        self.length = remaining
