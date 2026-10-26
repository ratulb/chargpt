"""Test: cached generation must equal uncached generation.

The cache is an optimization, not a semantics change. For the same prompt,
same seed, same model weights, and same sampling parameters, `generate` and
`generate_cached` must produce byte-identical output. If they diverge, the
cache is wrong — and the first differing token localizes the bug.

Four cases exercise the four paths in `generate_cached`:
  1. no eviction      (n < seq_length)
  2. eviction         (n > seq_length, window slides repeatedly)
  3. prompt length 1  (degenerate prompt, cache starts with 1 slot)
  4. prompt > window  (prefill crops to the last seq_length tokens)

Each case runs with two seeds to catch seed-handling bugs.
"""

from std.testing import assert_true, TestSuite
from std.sys import argv

from tenmo.tensor import Tensor
from tenmo.adamw import AdamW
from tenmo.shared.indexhelper import i as Indexer, s as Slicer
from bpe import BPETokenizer
from bpe.pretokenizer import PreTokenizer, ByteMapping, utf8_codepoint_byte_length

from chargpt import (
    CharGPT,
    CharPretokenizer,
    Dataset,
    INT,
    INT64,
    F32,
    HParams,
)
from shared import CmdLineArgs, TinyShakespeare


# ---- helpers ----------------------------------------------------------------

def make_model(h_params: HParams, vocab_size: Int) -> CharGPT:
    var model = CharGPT(vocab_size, h_params)
    return model^


def make_small_dataset(h_params: HParams, vocab_size: Int) raises -> Dataset:
    # A short synthetic corpus: enough for a few training steps, not enough
    # to matter for the test's purpose (non-trivial weights).
    var text = String(
        "To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles "
    )
    var tokenizer = BPETokenizer[CharPretokenizer]()
    var corpus = Span[String]([text])
    tokenizer.train(corpus, 300)
    var data = Tensor[INT].d1(tokenizer.encode(text))
    return Dataset(data^, h_params)


def train_a_few_steps(
    mut model: CharGPT,
    ref dataset: Dataset,
    mut optimizer: AdamW[F32],
    steps: Int,
) raises:
    # Enough steps that the weights are not the initial kaiming noise, so
    # the comparison is not testing "two identical random models."
    model.train()
    for _ in range(steps):
        var (xb, yb) = dataset.random_batch("train")
        var output = model(xb, yb)
        ref loss = output[1]
        ref loss_inner = loss.value()
        optimizer.zero_grad()
        loss_inner.backward()
        optimizer.step()
    model.eval()


def make_prompt(length: Int, vocab_size: Int) -> Tensor[INT]:
    # A deterministic prompt: ids 1..length, all valid (< vocab_size).
    var ids = List[Scalar[INT]](capacity=length)
    for k in range(length):
        ids.append(Scalar[INT](1 + k % (vocab_size - 1)))
    return Tensor[INT].d1(ids^).reshape(1, length)


def assert_same_ids(
    a: Tensor[INT],
    b: Tensor[INT],
    case_name: String,
    seed: Int,
) raises:
    # Compare element by element so a failure reports the first divergence.
    var la = a.numels()
    var lb = b.numels()
    assert_true(la == lb)
    for k in range(la):
        var va = Int(a[Slicer(), Indexer(k)].item()) if a.rank() == 2 else Int(a[Indexer(k)].item())
        var vb = Int(b[Slicer(), Indexer(k)].item()) if b.rank() == 2 else Int(b[Indexer(k)].item())
        if va != vb:
            print(
                "DIVERGED [", case_name, ", seed=", seed, "] at token ",
                k, ": uncached=", va, " cached=", vb,
            )
            assert_true(False)
    print("  ok: ", case_name, " seed=", seed, " (", la, " tokens)")


# ---- the four cases ---------------------------------------------------------

def test_case_no_eviction() raises:
    # seq_length = 32, generate 20 < 32 tokens: no eviction ever happens.
    var h_params = HParams(argv())
    h_params.seq_length = 32
    h_params.max_new_tokens = 20

    var tokenizer = BPETokenizer[CharPretokenizer]()
    var dataset = make_small_dataset(h_params, 256)
    var model = make_model(h_params, 256)
    var optimizer = AdamW[F32](
        model.parameters(),
        lr=Scalar[F32](1e-3),
    )
    train_a_few_steps(model, dataset, optimizer, 20)

    var prompt = make_prompt(5, 256)  # length 5
    for seed in [7, 42]:
        var out_u = model.generate(prompt, 20, init_seed=Optional[Int](seed))
        var out_c = model.generate_cached(prompt, 20, init_seed=Optional[Int](seed))
        assert_same_ids(out_u, out_c, "no_eviction", seed)
    print("test_case_no_eviction: ok")


def test_case_eviction() raises:
    # seq_length = 16, generate 40 > 16 tokens: the window slides
    # repeatedly, exercising evict_oldest on every step after the 16th.
    var h_params = HParams(argv())
    h_params.seq_length = 16
    h_params.max_new_tokens = 40

    var dataset = make_small_dataset(h_params, 256)
    var model = make_model(h_params, 256)
    var optimizer = AdamW[F32](model.parameters(), lr=Scalar[F32](1e-3))
    train_a_few_steps(model, dataset, optimizer, 20)

    var prompt = make_prompt(5, 256)
    for seed in [7, 42]:
        var out_u = model.generate(prompt, 40, init_seed=Optional[Int](seed))
        var out_c = model.generate_cached(prompt, 40, init_seed=Optional[Int](seed))
        assert_same_ids(out_u, out_c, "eviction", seed)
    print("test_case_eviction: ok")


def test_case_prompt_length_1() raises:
    # Prompt of length 1: the degenerate case where prefill feeds a
    # single token and the cache starts with one slot.
    var h_params = HParams(argv())
    h_params.seq_length = 32
    h_params.max_new_tokens = 10

    var dataset = make_small_dataset(h_params, 256)
    var model = make_model(h_params, 256)
    var optimizer = AdamW[F32](model.parameters(), lr=Scalar[F32](1e-3))
    train_a_few_steps(model, dataset, optimizer, 20)

    var prompt = make_prompt(1, 256)  # length 1
    for seed in [7, 42]:
        var out_u = model.generate(prompt, 10, init_seed=Optional[Int](seed))
        var out_c = model.generate_cached(prompt, 10, init_seed=Optional[Int](seed))
        assert_same_ids(out_u, out_c, "prompt_len_1", seed)
    print("test_case_prompt_length_1: ok")


def test_case_prompt_longer_than_window() raises:
    # Prompt of length 48 > seq_length 32: prefill must crop to the
    # last 32 tokens, and decode must continue from there.
    var h_params = HParams(argv())
    h_params.seq_length = 32
    h_params.max_new_tokens = 10

    var dataset = make_small_dataset(h_params, 256)
    var model = make_model(h_params, 256)
    var optimizer = AdamW[F32](model.parameters(), lr=Scalar[F32](1e-3))
    train_a_few_steps(model, dataset, optimizer, 20)

    var prompt = make_prompt(48, 256)  # length 48 > seq_length 32
    for seed in [7, 42]:
        var out_u = model.generate(prompt, 10, init_seed=Optional[Int](seed))
        var out_c = model.generate_cached(prompt, 10, init_seed=Optional[Int](seed))
        assert_same_ids(out_u, out_c, "prompt_gt_window", seed)
    print("test_case_prompt_longer_than_window: ok")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
