from std.testing import assert_true, TestSuite
from chargpt import MultiHeadAttention
from tenmo.tensor import Tensor
from kvcache import KVCache

comptime F32 = DType.float32


def test_prefill_equals_nocache() raises:
    var attn = MultiHeadAttention(
        n_embd=8, n_head=2, seq_length=8, drop_ratio=0.0
    )
    attn.eval()  # dropout off
    var x = Tensor[F32].rand(1, 4, 8)  # (B=1, T=4, C=8)
    # Uncached reference:
    var no_cache_out = attn(x)
    # Cached prefill: fresh cache, feed the whole x once.
    var cache = KVCache[F32](1, 2, 8, 4)  # capacity 8, head_dim 4
    var cache_out = attn(x, cache)
    # After prefill, cache.length == 4 (the whole prompt).
    assert_true(cache.length == 4)
    # Outputs must match: same Q, same K/V (cache holds exactly x's K/V),
    # same mask (tril[:4, :4] in both).
    assert_true(no_cache_out == cache_out)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
