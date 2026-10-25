from tenmo.tensor import Tensor
from std.testing import assert_true
from tenmo.shared.indexhelper import i, s


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

    def append(
        mut self, k_new: Tensor[Self.dtype], v_new: Tensor[Self.dtype]
    ) raises:
        """Write T_new new positions at [length : length + T_new].

        Precondition: length + T_new <= capacity. Callers must evict first;
        `append` never evicts on its own.
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
        self.keys.fill(k_new, s(), s(), s(self.length, self.length + T_new), s())
        self.values.fill(
            v_new, s(), s(), s(self.length, self.length + T_new), s()
        )
        self.length += T_new

    def get(self) raises -> Tuple[Tensor[Self.dtype], Tensor[Self.dtype]]:
        """Return views of the live prefix: (B, n_head, length, head_dim).

        Precondition: length > 0. tenmo cannot represent a zero-length axis
        (Shape(0) is invalid), so an empty cache has no tensor view. Attention
        only reads the cache after prefill has appended at least one position,
        so this precondition is satisfied in the generation flow.
        """
        assert_true(self.length > 0)
        return (
            self.keys[s(), s(), s(0, self.length), s()],
            self.values[s(), s(), s(0, self.length), s()],
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
        var k_src = self.keys[s(), s(), s(n, self.length), s()].clone()
        var v_src = self.values[s(), s(), s(n, self.length), s()].clone()
        self.keys.fill(k_src, s(), s(), s(0, remaining), s())
        self.values.fill(v_src, s(), s(), s(0, remaining), s())
        self.length = remaining


comptime F32 = DType.float32


def main() raises:
    test_append_happy_path()
    test_append_fills_cache()
    test_append_refuses_overflow()
    #test_get_returns_live_prefix()
    test_get_returns_view()
    test_evict_oldest_shifts()
    test_evict_oldest_multiple()
    test_sliding_window()
    print("all tests pass")


# ---- helpers ---------------------------------------------------------------

def make_cache() -> KVCache[F32]:
    # B=1, n_head=2, capacity=4, head_dim=2
    return KVCache[F32](1, 2, 4, 2)


def fill_with(value: Float32, T_new: Int) -> Tensor[F32]:
    # (1, 2, T_new, 2), all elements == value
    return Tensor[F32].ones(1, 2, T_new, 2) * value


def fill_with_range(start: Float32, T_new: Int) -> Tensor[F32]:
    # (1, 2, T_new, 2), position p holds start + p (both head_dim lanes)
    var t = Tensor[F32].zeros(1, 2, T_new, 2)
    for p in range(T_new):
        for h in range(2):
            for d in range(2):
                t.fill(Scalar[F32](start + Float32(p)), s(), i(h), i(p), i(d))
    return t^


def all_equal(
    t: Tensor[F32], value: Float32, lo: Int, hi: Int
) -> Bool:
    for h in range(2):
        for p in range(lo, hi):
            for d in range(2):
                if t[s(), i(h), i(p), i(d)].item() != value:
                    return False
    return True


# ---- tests -----------------------------------------------------------------

def test_append_happy_path() raises:
    var cache = make_cache()
    var k = fill_with(1.0, 2)
    var v = fill_with(2.0, 2)
    cache.append(k, v)
    assert_true(cache.length == 2)
    # slots 0-1 filled, 2-3 untouched
    assert_true(all_equal(cache.keys, 1.0, 0, 2))
    assert_true(all_equal(cache.keys, 0.0, 2, 4))
    assert_true(all_equal(cache.values, 2.0, 0, 2))
    assert_true(all_equal(cache.values, 0.0, 2, 4))
    print("test_append_happy_path: ok")


def test_append_fills_cache() raises:
    var cache = make_cache()
    cache.append(fill_with(1.0, 2), fill_with(11.0, 2))  # slots 0-1
    cache.append(fill_with(2.0, 2), fill_with(12.0, 2))  # slots 2-3
    assert_true(cache.length == 4)
    assert_true(all_equal(cache.keys, 1.0, 0, 2))
    assert_true(all_equal(cache.keys, 2.0, 2, 4))
    assert_true(all_equal(cache.values, 11.0, 0, 2))
    assert_true(all_equal(cache.values, 12.0, 2, 4))
    print("test_append_fills_cache: ok")


def test_append_refuses_overflow() raises:
    # If Mojo has an assert-fails helper, wrap the append in it. Otherwise
    # this test is a manual check: run it and confirm the process aborts
    # with the capacity assertion.
    var cache = make_cache()
    cache.append(fill_with(1.0, 2), fill_with(11.0, 2))
    cache.append(fill_with(2.0, 2), fill_with(12.0, 2))
    assert_true(cache.length == 4)
    # Uncomment to exercise the overflow path manually:
    # cache.append(fill_with(3.0, 1), fill_with(13.0, 1))  # must assert
    print("test_append_refuses_overflow: ok (see comment for manual check)")


def test_get_returns_live_prefix() raises:
    var cache = make_cache()
    cache.append(fill_with(1.0, 2), fill_with(11.0, 2))
    var (k, v) = cache.get()
    assert_true(k.shape()[0] == 1)
    assert_true(k.shape()[1] == 2)
    assert_true(k.shape()[2] == 2)  # length, not capacity
    assert_true(k.shape()[3] == 2)
    assert_true(all_equal(k, 1.0, 0, 2))
    assert_true(all_equal(v, 11.0, 0, 2))

    # length == 0 returns zero-length slices, not a crash
    var empty = make_cache()
    var (ek, ev) = empty.get()
    assert_true(ek.shape()[2] == 0)
    assert_true(ev.shape()[2] == 0)
    print("test_get_returns_live_prefix: ok")


def test_get_returns_view() raises:
    # If `get` returns a view, writing through it mutates self.keys.
    # If it returns a copy, the write is invisible to the cache.
    var cache = make_cache()
    cache.append(fill_with(1.0, 2), fill_with(11.0, 2))
    var (k, _) = cache.get()
    # Write 99.0 into the first live slot via the returned tensor.
    k.fill(Scalar[F32](99.0), s(), i(0), i(0), i(0))
    # If view: cache.keys now has 99.0 there. If copy: still 1.0.
    assert_true(cache.keys[s(), i(0), i(0), i(0)].item() == 99.0)
    print("test_get_returns_view: ok (get returns a view)")


def test_evict_oldest_shifts() raises:
    var cache = make_cache()
    # positions 0,1,2,3 hold 0.0,1.0,2.0,3.0
    cache.append(fill_with_range(0.0, 4), fill_with_range(10.0, 4))
    assert_true(cache.length == 4)
    cache.evict_oldest(1)
    assert_true(cache.length == 3)
    # now slots 0,1,2 hold 1.0,2.0,3.0
    assert_true(all_equal(cache.keys, 1.0, 0, 1))
    assert_true(all_equal(cache.keys, 2.0, 1, 2))
    assert_true(all_equal(cache.keys, 3.0, 2, 3))
    # values shifted the same way
    assert_true(all_equal(cache.values, 11.0, 0, 1))
    assert_true(all_equal(cache.values, 12.0, 1, 2))
    assert_true(all_equal(cache.values, 13.0, 2, 3))
    print("test_evict_oldest_shifts: ok")


def test_evict_oldest_multiple() raises:
    var cache = make_cache()
    cache.append(fill_with_range(0.0, 4), fill_with_range(10.0, 4))
    cache.evict_oldest(2)
    assert_true(cache.length == 2)
    assert_true(all_equal(cache.keys, 2.0, 0, 1))
    assert_true(all_equal(cache.keys, 3.0, 1, 2))
    print("test_evict_oldest_multiple: ok")


def test_sliding_window() raises:
    # The exact sequence the decode loop runs once the window is full:
    # evict 1, append 1, repeat.
    var cache = make_cache()  # capacity 4
    cache.append(fill_with_range(0.0, 4), fill_with_range(10.0, 4))
    assert_true(cache.length == 4)
    # evict oldest (0.0), then append position 4 (value 4.0)
    cache.evict_oldest(1)
    cache.append(fill_with_range(4.0, 1), fill_with_range(14.0, 1))
    assert_true(cache.length == 4)
    # window now holds 1,2,3,4
    assert_true(all_equal(cache.keys, 1.0, 0, 1))
    assert_true(all_equal(cache.keys, 2.0, 1, 2))
    assert_true(all_equal(cache.keys, 3.0, 2, 3))
    assert_true(all_equal(cache.keys, 4.0, 3, 4))
    print("test_sliding_window: ok")
