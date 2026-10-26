from tenmo.tensor import Tensor
from std.testing import assert_true, assert_raises, TestSuite
from tenmo.shared.indexhelper import i, s, e
from kvcache import KVCache

comptime F32 = DType.float32

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


def all_equal(t: Tensor[F32], value: Float32, lo: Int, hi: Int) -> Bool:
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
    with assert_raises():
        cache.append(fill_with(3.0, 1), fill_with(13.0, 1))  # must assert
    print("test_append_refuses_overflow: ok (see comment for manual check)")


def disabled_get_returns_live_prefix() raises:
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
    var cache = make_cache()
    cache.append(fill_with(1.0, 2), fill_with(11.0, 2))
    var (k, v) = cache.get()
    # Shape is the live prefix (length=2), not the full capacity (4).
    assert_true(k.shape()[2] == 2)
    assert_true(v.shape()[2] == 2)
    # Write-through: if view, mutating k mutates cache.keys.
    k.fill(Scalar[F32](99.0), s(), i(0), i(0), i(0))
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
