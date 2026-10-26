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


def test_decode_equals_nocache_on_the_extended_sequence() raises:
    var cache = KVCache[F32](1, 2, 8, 4)  # capacity 8, head_dim 4
    var attn = MultiHeadAttention(
        n_embd=8, n_head=2, seq_length=8, drop_ratio=0.0
    )
    attn.eval()  # dropout off
    var x = Tensor[F32].rand(1, 4, 8)  # (B=1, T=4, C=8)
    # Uncached reference:
    var no_cache_out = attn(x)
    # Cached prefill: fresh cache, feed the whole x once.
    var cache_out = attn(x, cache)

    # Now feed one new token and compare against an uncached forward over
    # the concatenated sequence.
    var x_new = Tensor[F32].rand(1, 1, 8)  # (B=1, T_new=1, C=8)
    var out_decode = attn(x_new, cache)  # cache.length -> 5
    assert_true(cache.length == 5)
    # Uncached reference: forward over the full length-5 sequence.
    var x_full = Tensor[F32].cat((x, x_new), axis=1)  # (1, 5, 8)
    var ref_full = attn(x_full)  # (1, 5, 8)
    # The decode output is the last position of the full-sequence output.
    var ref_last = ref_full[:, -1:, :]  # (1, 1, 8)
    assert_true(out_decode == ref_last)


def test_full_sequence_token_by_token() raises:
    var cache = KVCache[F32](1, 2, 8, 4)  # capacity 8, head_dim 4
    var attn = MultiHeadAttention(
        n_embd=8, n_head=2, seq_length=8, drop_ratio=0.0
    )
    attn.eval()  # dropout off

    # Full sequence: 8 tokens. Prefill the first 4, decode the rest.
    var x_full = Tensor[F32].rand(1, 8, 8)  # (B=1, T=8, C=8)
    var x_first_4 = x_full[:, 0:4, :]  # (1, 4, 8)
    var x_rest = x_full[:, 4:8, :]  # (1, 4, 8)

    # Prefill: feed the first 4 in one call. cache.length -> 4.
    var out_first = attn(x_first_4, cache)
    assert_true(cache.length == 4)

    # Decode: feed the remaining 4 one token at a time. cache.length
    # grows 5, 6, 7, 8. Collect each (1, 1, 8) output.
    var decoded = List[Tensor[F32]](capacity=4)
    for t in range(4):
        var x_t = x_rest[:, t : t + 1, :]  # (1, 1, 8)
        decoded.append(attn(x_t, cache))
        assert_true(cache.length == 5 + t)

    # Concatenate the prefill output (4 tokens) and the 4 decode outputs
    # -> (1, 8, 8), the cached forward over the whole sequence.
    var out_cached = Tensor[F32].cat(
        (out_first, decoded[0], decoded[1], decoded[2], decoded[3]),
        axis=1,
    )
    assert_true(out_cached.shape()[1] == 8)

    # Uncached reference: one forward over the full 8-token sequence.
    var out_ref = attn(x_full)  # (1, 8, 8)

    # Every position's output must match. If the cache is wrong (stale K/V,
    # wrong mask on a decode step, off-by-one in length), the divergence
    # shows up here — typically at the first decode step.
    assert_true(out_cached == out_ref)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
