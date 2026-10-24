# =============================================================================
# bigram.mojo - step 1: the simplest possible language model.
#
# A bigram model predicts the next character from the CURRENT character only.
# It has no context and no attention: just a table with one row per character,
# and that row holds the scores for what comes next. "After 'q', 'u' is very
# likely" is exactly the kind of thing it can learn, and nothing more.
#
# It is the baseline for chargpt.mojo. Everything chargpt adds (embeddings of
# position, attention, blocks) is an attempt to do better than this table.
#
# Reading guide:
#   1. CharTokenizer   characters <-> integer ids (65 of them, built by hand)
#   2. BigramModel     the table, the loss, and `generate`
#   3. Dataset         random (input, target) windows
#   4. Loss.estimate   a steadier loss reading than one batch gives
#   5. main            data, model, optimizer, training loop
#
# Run it (settings are key=value arguments, see shared.mojo):
#
#   mojo bigram.mojo
#   mojo bigram.mojo max_iters=5000 learning_rate=0.01
#
# Shape notation: B = batch size, T = sequence length, V = vocab size.
# Here C (the last axis of the logits) is also V, because the table's rows
# are scores over the vocabulary.
# =============================================================================

from tenmo.tensor import Tensor
from std.pathlib import Path
from std.os.process import Process
from std.collections import Set
from std.sys import argv
from tenmo.shared.indexhelper import i as Indexer, s as Slicer
from tenmo.embedding import Embedding
from tenmo.crossentropy import CrossEntropyLoss
from std.random import random_ui64, seed
from tenmo.adamw import AdamW

from shared import CmdLineArgs, TinyShakespeare

# Short names for the element types: INT for token ids, INT64 for what the
# embedding lookup expects, F32 for weights and activations.
comptime INT = DType.int
comptime INT64 = DType.int64
comptime F32 = DType.float32

# The run-time settings live in `CmdLineArgs` (shared.mojo).
comptime HParams = CmdLineArgs


# -----------------------------------------------------------------------------
# 1. CharTokenizer: the hand-made version
# -----------------------------------------------------------------------------
# Collect the distinct bytes that occur in the text, sort them, and number
# them 0, 1, 2, ... That gives about 65 ids for Tiny Shakespeare. This is the
# whole idea of tokenisation in its simplest form: a lookup in each direction.
#
# chargpt.mojo does the same job with mbpe instead, and gets 256 ids (one per
# possible byte value) rather than only the ones that happen to appear.
struct CharTokenizer(Sized):
    var stoi: Dict[String, Int]  # "string to int": character -> id
    var itos: Dict[Int, String]  # "int to string": id -> character

    def __init__(out self, var text: String):
        # Distinct bytes in the text, sorted, so ids are the same every run.
        var byte_set = Set[Byte]([byte for byte in text.bytes()])
        var ints_from_bytes = [Int(byte) for byte in byte_set]
        sort(ints_from_bytes)
        # One one-character string per distinct byte.
        var chars_as_string = [chr(each) for each in ints_from_bytes]
        var vocab = "".join(chars_as_string)
        print("Vocabulary: ", vocab)
        # Position in the sorted list is the id.
        self.stoi = {s: i for i, s in enumerate(chars_as_string)}
        self.itos = {i: s for i, s in enumerate(chars_as_string)}

    def encode(self, s: String) raises -> List[Int]:
        # "hi" -> [id of 'h', id of 'i']
        return [self.stoi[chr(Int(byte))] for byte in s.bytes()]

    def decode(self, tokens: List[Int]) raises -> String:
        # The reverse: ids -> characters, joined into one string.
        return "".join([self.itos[token] for token in tokens])

    def __len__(self) -> Int:
        return len(self.itos)


# -----------------------------------------------------------------------------
# 2. BigramModel: a V x V table
# -----------------------------------------------------------------------------
# `Embedding(vocab_size, vocab_size)` is a table with one row per character.
# Looking up the row for the current character gives V scores (logits), one
# for each character that could come next. Training adjusts the table so that
# the right next character gets a high score.
#
# The whole model is V * V numbers (about 65 * 65 = 4,225).
struct BigramModel:
    var embedding_table: Embedding[F32]
    var criterion: CrossEntropyLoss[F32]

    def __init__(out self, vocab_size: Int):
        self.embedding_table = Embedding[F32](vocab_size, vocab_size)
        self.criterion = CrossEntropyLoss[F32]()

    def __call__(
        mut self,
        indices: Tensor[INT],
        targets: Optional[Tensor[INT]] = None,
    ) raises -> Tuple[Tensor[F32], Optional[Tensor[F32]]]:
        # indices (B, T) -> logits (B, T, V): one row looked up per character.
        # Returns (logits, loss); the loss is None unless targets are given.
        var logits = self.embedding_table(indices.to_dtype[INT64]())
        var loss: Optional[Tensor[F32]] = None
        if targets:
            # CrossEntropyLoss logits and target conformation
            # It wants one row of scores per prediction, so flatten the batch
            # and time axes together: (B, T, V) -> (B*T, V), targets (B*T).
            # Cross-entropy is -log(probability given to the true next
            # character), averaged over all B*T predictions. A model that
            # knows nothing scores ln(65) = 4.17.
            ref logits_shape = logits.shape()
            var B, T, C = logits_shape[0], logits_shape[1], logits_shape[2]
            loss = self.criterion(
                logits.reshape(B * T, C),
                targets.value().to_dtype[INT64]().reshape(B * T),
            )
        return logits^, loss^

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        # The only weights are the table itself.
        return self.embedding_table.parameters()

    # No dropout or other mode-dependent layer here, so the flag changes
    # nothing in practice. It is kept so this model has the same train/eval
    # interface as CharGPT.
    def train(mut self):
        self.embedding_table.training = True

    def eval(mut self):
        self.embedding_table.training = False

    def generate(
        mut self,
        indices: Tensor[INT],
        max_new_tokens: Int,
        max_seed: UInt64,
    ) raises -> Tensor[INT]:
        # Write text one character at a time: look up the scores for the
        # latest character, turn them into probabilities, draw one character
        # at random from those probabilities, append it, repeat.
        # indices (B, T) from current context
        var appended = indices.clone()
        for _ in range(max_new_tokens):
            # Logits for every position of the text so far: (B, T, V).
            var (logits, _) = self(appended)
            # logits now (B, C)
            # Only the LAST position matters: it is the prediction for the
            # character that has not been written yet.
            logits = logits[Slicer(), -1, Slicer()]
            # Softmax turns scores into probabilities that sum to 1.
            var probas = logits.softmax[track_grad=False](axes=[-1])
            # next_id's shape (B, 1)
            # `multinomial` draws one id according to those probabilities.
            # It needs a seed, so we re-seed and pass a fresh random one.
            seed()
            var next_id = probas.multinomial[index_dtype=INT](
                init_seed=Int(random_ui64(0, max_seed))
            )
            # Append sampled token to the running sequence.
            appended = Tensor[INT].concat[track_grad=False](
                [appended, next_id], axis=1
            )
        return appended^


# -----------------------------------------------------------------------------
# 3. Dataset: turning one long id sequence into training examples
# -----------------------------------------------------------------------------
# A training example is a random window of `seq_length` ids (x) and the same
# window shifted one place to the right (y). At every position, y holds the
# character that follows:
#
#   text:  T  o  _  b  e  _  o  r
#   x:     T  o  _  b  e
#   y:     o  _  b  e  _
#
# For a bigram model the window length does not change what is learnt (each
# prediction uses only one character), it just packs more predictions into a
# batch. It starts to matter in chargpt.mojo, where the model looks back over
# the whole window.
struct Dataset:
    var train_data: Tensor[INT]
    var val_data: Tensor[INT]
    var batch_size: Int
    var seq_length: Int

    def __init__(
        out self,
        var data: Tensor[INT],
        h_params: HParams,
        split_ratio: Float64 = 0.9,
    ):
        # First 90% of the text trains the model; the last 10% is held back
        # for validation and never used to update a weight.
        var split = Int(split_ratio * Float64(len(data)))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.batch_size = h_params.batch_size
        self.seq_length = h_params.seq_length

    def random_batch(
        self, mode: String  # "train" or "val": which part of the text to use
    ) raises -> Tuple[Tensor[INT], Tensor[INT]]:
        ref data_src = self.train_data if mode == "train" else self.val_data
        # `batch_size` random start positions; `high` leaves room for the
        # window plus the one extra character the targets need.
        var indices = Tensor[INT].rand(
            self.batch_size, high=len(data_src) - self.seq_length
        )
        # x: (B, T) the windows;  y: (B, T) the same windows shifted by one.
        var x = Tensor[INT].stack(
            [data_src[idx : idx + self.seq_length] for _, idx in indices]
        )
        var y = Tensor[INT].stack(
            [
                data_src[idx + 1 : idx + self.seq_length + 1]
                for _, idx in indices
            ]
        )
        return x^, y^


# -----------------------------------------------------------------------------
# 4. Loss.estimate: a steadier loss reading
# -----------------------------------------------------------------------------
# The loss of a single random batch is noisy. Averaging `eval_iters` batches
# from each split gives a number that is comparable from one evaluation to the
# next. Train and val should stay close together; a bigram table is far too
# small to memorise the text.
struct Loss:
    @staticmethod
    def estimate(
        mut model: BigramModel,
        ref dataset: Dataset,
        eval_iters: Int,
    ) raises -> Dict[String, Float32]:
        model.eval()
        var out = {"train": Float32(0.0), "val": Float32(0.0)}
        for mode in ["train", "val"]:
            var losses = Tensor[F32].zeros(eval_iters)
            for k in range(eval_iters):
                var (x, y) = dataset.random_batch(mode)
                var (_, loss) = model(x, y)
                # loss.value() because loss is Optional
                losses[k] = loss.value().item()
            out[mode] = losses.mean().item()
        model.train()
        return out^


# -----------------------------------------------------------------------------
# 5. main: put it all together
# -----------------------------------------------------------------------------
def main() raises:
    # Settings come from key=value arguments; an unknown key or an unparsable
    # value is silently ignored, so check the "Hyper params" line below.
    var h_params = CmdLineArgs(argv())
    print("Hyper params: ", h_params)
    var text = TinyShakespeare.text()
    var tokenizer = CharTokenizer(text)

    var vocab_size = len(tokenizer)
    print("Tokenizer len: ", len(tokenizer))
    # The whole text as one long tensor of ids.
    var data = Tensor[INT].d1(tokenizer.encode(text^))
    var dataset = Dataset(data^, h_params)
    var model = BigramModel(vocab_size)

    # Text from the untrained model. The start "prompt" is a single id 0, the
    # first character in sorted order (the newline). Expect pure noise.
    print(
        tokenizer.decode(
            model.generate(
                Tensor[INT].zeros(1, 1),
                max_new_tokens=h_params.max_new_tokens,
                max_seed=h_params.seed_range,
            )[Indexer(0), Slicer()].tolist()
        )
    )

    # AdamW: gradient descent that adapts the step size of every weight.
    var optimizer = AdamW[F32](
        model.parameters(),
        lr=Scalar[F32](h_params.learning_rate),
    )

    # The training loop. One step = one batch:
    #   forward -> loss -> clear old gradients -> backward -> weight update
    for iter in range(h_params.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if (iter + 1) % h_params.eval_interval == 0:
            var losses = Loss.estimate(model, dataset, h_params.eval_iters)
            print(
                "Step {0}: train loss {1}, val loss {2}".format(
                    iter + 1, losses["train"], losses["val"]
                )
            )

        var (xb, yb) = dataset.random_batch("train")
        var output = model(xb, yb)
        ref _logits = output[0]
        ref loss = output[1]
        # loss is returned as Optional - hence `value()`
        ref loss_inner = loss.value()
        optimizer.zero_grad()  # gradients accumulate, so clear the last step's
        loss_inner.backward()  # how should each weight change to lower loss?
        optimizer.step()  # nudge every weight a little in that direction

        # Print a sample every `gen_interval` steps, to watch the text improve.
        if (iter + 1) % h_params.gen_interval == 0:
            print(
                tokenizer.decode(
                    model.generate(
                        Tensor[INT].zeros(1, 1),
                        max_new_tokens=h_params.max_new_tokens,
                        max_seed=h_params.seed_range,
                    )[Indexer(0), Slicer()].tolist()
                ),
                end="\n",
            )
