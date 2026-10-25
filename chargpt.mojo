"""A small character-level GPT, in Mojo, on CPU.

   Reader guidance - The file goes from the small pieces to the whole:

   CharPretokenizer     text -> one piece per character (mbpe plug-in)
   Dataset              ids  -> random (input, target) windows
   MultiHeadAttention   tokens look back at earlier tokens
   MLP                  each token thinks on its own
   Block                attention + MLP, each with a residual connection
   CharGPT              embeddings -> N blocks -> logits (and `generate`)
   Loss.estimate        a steadier loss reading than one batch gives
   generate_text        print a sample, starting from a validation prompt
   main                 tokenizer, data, model, optimizer, training loop

   Run it (settings are key=value arguments, see shared.mojo):

   mojo chargpt.mojo
   mojo chargpt.mojo batch_size=32 seq_length=32 max_iters=3000 n_embd=192

   Notation used in the shape comments below:
   B = batch size       (windows per step)
   T = sequence length  (characters per window, at most `seq_length`)
   C = n_embd           (width of every vector)
   V = vocab_size       (256 here: one id per byte value)
   """

from std.utils.numerics import neg_inf
from std.math import sqrt
from std.sys import argv

from tenmo.tensor import Tensor
from bpe import BPETokenizer
from bpe.pretokenizer import (
    PreTokenizer,
    ByteMapping,
    utf8_codepoint_byte_length,
)
from tenmo.shared.indexhelper import i as Indexer, s as Slicer
from tenmo.net import Linear
from tenmo.embedding import Embedding
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.adamw import AdamW
from tenmo.dropout import Dropout
from tenmo.net import GeLU
from tenmo.layernorm import LayerNorm
from tenmo.shared.shapes import Shape
from shared import CmdLineArgs, TinyShakespeare

# Short names for the element types used throughout.
# INT    token ids
# INT64  what the embedding lookup expects
# F32    all weights and activations

comptime INT = DType.int
comptime INT64 = DType.int64
comptime F32 = DType.float32

# The run-time settings (batch_size, seq_length, n_embd, ...) live in
# `CmdLineArgs`, in shared.mojo. This alias only gives them a friendlier name.
comptime HParams = CmdLineArgs


struct CharPretokenizer(PreTokenizer):
    """
    CharPretokenizer: what makes this a *character* model
    mbpe's BPE core knows nothing about words or characters. Before any merging,
    a pre-tokenizer cuts the text into pieces, and BPE only ever merges bytes
    *inside* one piece. Here every piece is a single character, so no merge can
    ever happen and the vocabulary stays at the 256 base byte values.
    """

    comptime byte_map: ByteMapping = ByteMapping.SEQUENTIAL

    def __init__(out self):
        pass

    @staticmethod
    def name() -> String:
        return String("char")

    def split[
        mut: Bool, //, origin: Origin[mut=mut]
    ](self, text: StringSpan[origin]) raises -> List[StringSpan[origin]]:
        var result = List[StringSpan[origin]]()
        var n = text.byte_length()
        if n == 0:
            return result^
        var span = text.as_bytes()
        var pos = 0
        while pos < n:
            # One "word" per codepoint. A UTF-8 character takes 1 to 4 bytes;
            # the first byte tells us how many. Clamp so a truncated trailing
            # sequence never reads past the end.
            var end = pos + utf8_codepoint_byte_length(span[pos])
            if end > n:
                end = n
            result.append(StringSpan(unsafe_from_utf8=span[pos:end]))
            pos = end
        return result^

    def write_to[T: Writer](self, mut writer: T):
        writer.write(String("CharPretokenizer"))


struct Dataset:
    """
    Dataset: turning one long id sequence into training examples
    The whole text is one long list of ids. A training example is a random
    window of `seq_length` ids (x) and the same window shifted one place to the
    right (y). So at every position, y holds the character that follows.

    text:  T  o  _  b  e  _  o  r
    x:     T  o  _  b  e            (what the model sees)
    y:     o  _  b  e  _            (what it must predict, position by position)

    One window therefore carries `seq_length` separate predictions.

    """

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
        # for validation and never used to update weights.
        var split = Int(split_ratio * Float64(len(data)))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.batch_size = h_params.batch_size
        self.seq_length = h_params.seq_length

    def random_batch(
        self,
        mode: String,  # "train" or "val": which part of the text to draw from
        batch_size_override: Optional[
            Int
        ] = None,  # Override batch size if need be. For example: Get initial prompt
        seq_length_override: Optional[
            Int
        ] = None,  # Override block size if needed. For example: For getting initail prompt
    ) raises -> Tuple[Tensor[INT], Tensor[INT]]:
        ref data_src = self.train_data if mode == "train" else self.val_data

        var size_of_batch = batch_size_override.or_else(self.batch_size)
        var size_of_seq = seq_length_override.or_else(self.seq_length)
        # `size_of_batch` random start positions. `high` leaves room for the
        # window plus the one extra character that the targets need.
        var indices = Tensor[INT].rand(
            size_of_batch, high=len(data_src) - size_of_seq
        )
        # In tenmo, iterating over a Tensor produces coord(s) and value(s).
        # We are interested in the values.
        var x = Tensor[INT].stack(
            [data_src[idx : idx + size_of_seq] for _, idx in indices]
        )
        var y = Tensor[INT].stack(
            [data_src[idx + 1 : idx + size_of_seq + 1] for _, idx in indices]
        )
        return x^, y^


# Multi-head causal self-attention: the decoder's "look back" block.
# Q, K, V come from one fused linear (c_attn); the lower-triangular mask
# (tril) hides future positions so token i attends only to tokens <= i.
struct MultiHeadAttention:
    """
    MultiHeadAttention: the "look back" step.

    Every position asks a question (query), every position offers a label (key)
    and some content (value). A position scores each earlier position by how
    well its query matches that position's key, turns the scores into weights
    with softmax, and takes the weighted average of the values.

    "Multi-head" means this is done n_head times in parallel, each head on its
    own slice of head_dim = n_embd / n_head numbers, so different heads can look
    for different things. "Causal" means a position may only look at itself and
    earlier positions; the lower-triangular mask (tril) enforces that.
    """

    var n_head: Int
    var head_dim: Int
    var c_attn: Linear[F32]  # Fused QKV: n_embd -> 3*n_embd
    var c_proj: Linear[F32]  # Output projection: mixes the heads back together
    var tril: Tensor[F32]  # (seq_length, seq_length) lower-triangular 1s
    var dropout: Dropout[F32]
    var inv_scale: Scalar[F32]  # 1/sqrt(head_dim), computed once

    def __init__(
        out self, n_embd: Int, n_head: Int, seq_length: Int, drop_ratio: Float32
    ):
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # One matrix produces queries, keys and values in a single multiply.
        self.c_attn = Linear[F32](
            in_features=n_embd,
            out_features=3 * n_embd,
            init_method="kaiming",
            bias=False,
        )
        self.c_proj = Linear[F32](
            in_features=n_embd, out_features=n_embd, init_method="kaiming"
        )
        # Row i of tril has ones in columns 0..i: "position i may look at
        # positions 0..i". Built once, then sliced to the current length.
        self.tril = Tensor[F32].ones(seq_length, seq_length).tril()
        self.dropout = Dropout[F32](drop_ratio)
        # Dividing the scores by sqrt(head_dim) keeps them from growing with
        # the head size, which would push softmax into a near one-hot spike.
        self.inv_scale = Scalar[F32](1.0 / sqrt(Float64(self.head_dim)))

    def __call__(mut self, x: Tensor[F32]) raises -> Tensor[F32]:
        # Causal multi-head attention forward. Input and output: (B, T, C).
        ref x_shape = x.shape()
        var B = x_shape[0]
        var T = x_shape[1]
        var C = x_shape[2]

        # Fused QKV projection, then split along feature axis.
        var qkv = self.c_attn(x)  # (B, T, 3*C)
        var q = qkv.slice(0, C, axis=2)  # (B, T, C)
        var k = qkv.slice(C, 2 * C, axis=2)  # (B, T, C)
        var v = qkv.slice(2 * C, 3 * C, axis=2)  # (B, T, C)
        # Split the C numbers of each token into n_head groups of head_dim,
        # and move the head axis forward so each head is a separate batch
        # entry for the matrix multiply:
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 1, 3])
        # For k we also swap T and head_dim in the same permute, giving
        # (B, n_head, head_dim, T). That is k already transposed, ready for
        # q @ k below, so no separate transpose is needed.
        k = k.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 3, 1])
        v = v.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 1, 3])
        # No need for transposing k, see comment above
        # scores[i][j] = how much position i's query matches position j's key.
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        var scaled_attn_scores = (
            q.matmul(k) * self.inv_scale
        )  # (B, n_head, T, T)

        # The mask goes in BEFORE softmax: every "future" cell (j > i) is set
        # to -infinity, so softmax gives it exactly zero weight. Zeroing the
        # weights afterwards would leave the rows not summing to 1.
        var masked_scores = scaled_attn_scores.masked_fill(
            self.tril[:T, :T] == 0, neg_inf[F32]()
        )
        # Softmax over the last axis: each row becomes weights over the
        # earlier positions that sum to 1.
        var attn_weights = masked_scores.softmax(axes=[-1])
        var attenuated_attn = self.dropout(attn_weights)
        # Weighted average of the values: (B, n_head, T, T) @ (B, n_head, T, head_dim)
        var weighted_values = attenuated_attn.matmul(
            v
        )  # (B, n_head, T, head_dim)
        # Put the heads back side by side: (B, n_head, T, hd) -> (B, T, C),
        # then let c_proj mix information across heads.
        var values = weighted_values.permute([0, 2, 1, 3]).reshape(B, T, C)
        return self.c_proj(values)

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        # The optimizer needs a list of every weight it should update.
        # `tril` is a constant mask, not a weight, so it is not included.
        var params = List[Pointer[Tensor[F32], MutAnyOrigin]]()
        params.extend(self.c_attn.parameters())
        params.extend(self.c_proj.parameters())
        return params^

    # train()/eval() flip the `training` flag on every layer that behaves
    # differently in the two modes (dropout in particular).
    def train(mut self):
        self.c_attn.training = True
        self.c_proj.training = True
        self.dropout.training = True

    def eval(mut self):
        self.c_attn.training = False
        self.c_proj.training = False
        self.dropout.training = False


struct MLP:
    """
    MLP: the "think" step
    Attention moves information *between* positions. The MLP then works on each
    position separately: widen to 4x, apply a smooth non-linearity (GELU), and
    narrow back. Without the non-linearity, two linear layers would collapse
    into one.
    """

    var c_fc: Linear[F32]  # n_embd -> 4*n_embd
    var gelu: GeLU[F32]  # 4*n_embd -> n_embd
    var c_proj: Linear[F32]
    var dropout: Dropout[F32]

    def __init__(out self, n_embd: Int, drop_ratio: Float32):
        self.c_fc = Linear[F32](
            in_features=n_embd, out_features=4 * n_embd, init_method="kaiming"
        )
        self.gelu = GeLU[F32]()
        self.c_proj = Linear[F32](
            in_features=4 * n_embd, out_features=n_embd, init_method="kaiming"
        )
        self.dropout = Dropout[F32](drop_ratio)

    def __call__(mut self, x: Tensor[F32]) -> Tensor[F32]:
        var h = self.c_fc(x)
        h = self.gelu(h)
        h = self.c_proj(h)
        return self.dropout(h)

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[F32], MutAnyOrigin]]()
        params.extend(self.c_fc.parameters())
        params.extend(self.c_proj.parameters())
        return params^

    def train(mut self):
        self.c_fc.training = True
        self.gelu.train()
        self.c_proj.training = True
        self.dropout.training = True

    def eval(mut self):
        self.c_fc.training = False
        self.gelu.eval()
        self.c_proj.training = False
        self.dropout.training = False


struct Block:
    """
    Block: one transformer layer
    Pre-norm layout: normalise, apply the sub-layer, and ADD the result to what
    came in (a residual connection). The residual gives gradients a direct path
    back through a deep stack, and lets each block make a small correction to
    the running representation rather than rebuild it.
    x -> LayerNorm -> attention -> (+ x) -> LayerNorm -> MLP -> (+ itself).

    """

    var ln1: LayerNorm[F32]
    var attn: MultiHeadAttention
    var ln2: LayerNorm[F32]
    var mlp: MLP

    def __init__(out self, h_params: HParams):
        self.ln1 = LayerNorm[F32](h_params.n_embd)
        self.attn = MultiHeadAttention(
            h_params.n_embd,
            h_params.n_head,
            h_params.seq_length,
            h_params.drop_ratio,
        )
        self.ln2 = LayerNorm[F32](h_params.n_embd)
        self.mlp = MLP(h_params.n_embd, h_params.drop_ratio)

    def __call__(mut self, x: Tensor[F32]) raises -> Tensor[F32]:
        var h = self.ln1(x)
        var a = self.attn(h)
        var residual = x + a
        var h2 = self.ln2(residual)
        var m = self.mlp(h2)
        return residual + m

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[F32], MutAnyOrigin]]()
        params.extend(self.ln1.parameters())
        params.extend(self.attn.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.mlp.parameters())
        return params^

    def train(mut self):
        self.ln1.train()
        self.attn.train()
        self.ln2.train()
        self.mlp.train()

    def eval(mut self):
        self.ln1.eval()
        self.attn.eval()
        self.ln2.eval()
        self.mlp.eval()


struct CharGPT:
    """
    CharGPT: the whole model
    ids (B, T) -> token + position embeddings -> n_layer blocks -> LayerNorm
         -> linear head -> logits (B, T, V).

    A logit is an unnormalised score. logits[b, t, :] holds one score for each
    of the V possible next characters, given everything up to position t.
    """

    var token_embedding_table: Embedding[F32]
    var position_embedding_table: Embedding[F32]
    var blocks: List[Block]
    var ln_f: LayerNorm[F32]
    var lm_head: Linear[F32]
    var criterion: CrossEntropyLoss[F32]
    var h_params: HParams

    def __init__(out self, vocab_size: Int, h_params: HParams):
        self.token_embedding_table = Embedding[F32](
            vocab_size, h_params.n_embd, init_method="kaiming"
        )
        # One learned vector per position 0..seq_length-1. This is also why
        # the model cannot take a context longer than seq_length(see `generate`).
        self.position_embedding_table = Embedding[F32](
            h_params.seq_length, h_params.n_embd, init_method="kaiming"
        )
        self.blocks = List[Block](capacity=h_params.n_layer)
        for _ in range(h_params.n_layer):
            self.blocks.append(Block(h_params))
        self.ln_f = LayerNorm[F32](h_params.n_embd)
        self.lm_head = Linear[F32](
            in_features=h_params.n_embd,
            out_features=vocab_size,
            init_method="kaiming",
        )
        self.criterion = CrossEntropyLoss[F32]()
        self.h_params = h_params

    def __call__(
        mut self, indices: Tensor[INT], targets: Optional[Tensor[INT]] = None
    ) raises -> Tuple[Tensor[F32], Optional[Tensor[F32]]]:
        # Forward pass of the decoder: embeddings -> blocks -> head.
        # Returns (logits, loss). The loss is None unless targets are given.
        ref indices_shape = indices.shape()
        var token_count = indices_shape[1]

        # Look up token and absolute position embeddings, then add (broadcasts).
        var token_embeddings = self.token_embedding_table(
            indices.to_dtype[INT64]()
        )  # (B, T, n_embd)
        var position_embeddings = self.position_embedding_table(
            Tensor[INT64].arange(Int64(token_count))
        )  # (T, n_embd) broadcasts to (B, T, n_embd)
        # Attention itself has no sense of order; the position vectors are
        # what tell the model "this is the 3rd character, that is the 7th".
        var x = token_embeddings + position_embeddings

        # Run through N pre-norm transformer blocks.
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            x = block(x)

        # Final LayerNorm then linear projection to vocabulary logits.
        x = self.ln_f(x)
        var logits = self.lm_head(x)  # (B, T, vocab_size)
        var loss: Optional[Tensor[F32]] = None

        # Training: flatten (B,T) -> (B*T) and compute cross-entropy vs targets.
        # Cross-entropy is -log(probability given to the true next character),
        # averaged over all B*T predictions. Each prediction is a row of
        # V scores with one correct answer, hence the flattening.
        if targets:
            ref logits_shape = logits.shape()
            var B, T, C = logits_shape[0], logits_shape[1], logits_shape[2]
            loss = self.criterion(
                logits.reshape(B * T, C),
                targets.value().to_dtype[INT64]().reshape(B * T),
            )

        return logits^, loss^

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        # Collect every weight in the model, for the optimizer.
        var params = List[Pointer[Tensor[F32], MutAnyOrigin]]()
        params.extend(self.token_embedding_table.parameters())
        params.extend(self.position_embedding_table.parameters())

        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            ref block_params = block.parameters()
            for param in range(len(block_params)):
                params.append(block_params[param])
        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())

        return params^

    def train(mut self):
        self.token_embedding_table.training = True
        self.position_embedding_table.training = True

        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            block.train()

        self.ln_f.train()
        self.lm_head.training = True

    def eval(mut self):
        self.token_embedding_table.training = False
        self.position_embedding_table.training = False

        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            block.eval()

        self.ln_f.eval()
        self.lm_head.training = False

    def generate(
        mut self,
        indices: Tensor[INT],
        max_new_tokens: Int,
        temperature: Float32 = 0.8,
        top_k: Int = 50,
        end_id: Int = -1,
        init_seed: Optional[Int] = None,
    ) raises -> Tensor[INT]:
        """Uncached autoregressive generation: prompt + up to `max_new_tokens`.

        Each step runs the full model over the last-`seq_length` window, samples
        the final position, appends. Stops early on `end_id` (>= 0).

        Seeding:
        - `init_seed == None` → every step passes `None` to `sample`, so each
          step (and each whole run) draws fresh entropy. Non-reproducible; the
          default.
        - `init_seed == Some(s)` → step `m` uses seed `s + m`, so a whole run is
          reproducible while each step's Gumbel noise is independent. `s + m`
          must not overflow `Int` (i.e. `s + max_new_tokens` must fit).

        Returns the id stream with the prompt prefix intact.
        """
        if indices.numels() == 0:
            raise Error("generate: empty prompt")
        if max_new_tokens <= 0:
            raise Error("generate: max_new_tokens must be >= 1")

        # eval() switches dropout off; train() at the end switches it back.
        # Forgetting that last call leaves training silently running in eval
        # mode after the first sample - this was a hard lesson. Forgot to set
        # train() back and loss went thru the roof.
        self.eval()

        # Copy the prompt ids into a growable list, validating as we go. On
        # the error path we must restore train() before raising, or the
        # caller's model is left in eval mode.
        var ids = List[Int](capacity=indices.numels() + max_new_tokens)
        for i in range(indices.numels()):
            var v = Int(indices.get(i))
            if v < 0:
                self.train()
                raise Error("generate: negative prompt id")
            ids.append(v)

        # `base_seed` is only read when `have_seed` is true; the `else 0` is a
        # placeholder so the variable is always initialized.
        var base_seed = init_seed.value() if init_seed else 0
        var have_seed = init_seed != None

        var made = 0
        while made < max_new_tokens:
            var total = len(ids)
            # Position embeddings only have `seq_length` rows, so the model
            # cannot accept a context longer than seq_length. Crop the growing
            # sequence to its last seq_length ids (like idx[:, -seq_length:]
            # in nanoGPT) before each forward pass.
            var T = min(total, self.h_params.seq_length)
            var window = List[Scalar[INT]](capacity=T)
            for i in range(total - T, total):
                window.append(Scalar[INT](ids[i]))
            var batch = Tensor[INT].from_list[INT](window^)
            var flat = batch.reshape(1, T)

            var (logits, _) = self(flat)  # (1, T, V), grad-free under eval()
            var V = logits.shape()[2]
            var last = logits[Slicer(), -1, Slicer()]  # (1, V)
            var last_1d = last.reshape(V)

            # Seeded run: step m uses `base_seed + m` so each step's Gumbel
            # noise is independent while the whole run stays reproducible.
            # Unseeded run: pass None through, so `multinomial` draws fresh
            # entropy per step.
            var step_seed = Optional[Int](None)
            if have_seed:
                step_seed = Optional[Int](base_seed + made)
            var next_id = Self.sample(
                last_1d, temperature, top_k, init_seed=step_seed
            )

            ids.append(next_id)
            made += 1
            if end_id >= 0 and next_id == end_id:
                break

        self.train()

        var out = List[Scalar[INT]](capacity=len(ids))
        for i in range(len(ids)):
            out.append(Scalar[INT](ids[i]))
        return Tensor[INT].from_list[INT](out^)

    @staticmethod
    def sample(
        logits: Tensor[F32],
        temperature: Float32 = 0.8,
        top_k: Int = 50,
        init_seed: Optional[Int] = None,
    ) raises -> Int:
        """Draw one token id from 1-D `logits` (V,).

        - `temperature <= 0` or `top_k == 1` → greedy argmax (deterministic).
        - Otherwise: keep the top-`k` entries (`0 < k < V`; `k <= 0` or `k >= V`
          disables the mask), scale the survivors by `1/temperature`, softmax,
          single `multinomial` draw.

        Seeding:
        - `init_seed == None` → `multinomial` draws fresh entropy per call, so
          repeated calls are (almost surely) different. This is the default.
        - `init_seed != None` → the draw is reproducible for a given `s`.

        Scaling by positive `1/temperature` is monotonic, so it commutes with
        top-k selection: we select on raw logits and scale only the survivors.
        """
        var V = logits.numels()
        if V == 0:
            raise Error("sample: empty logits")

        # temperature <= 0 means "no randomness"; top_k == 1 means "only one
        # candidate".
        if temperature <= 0.0 or top_k == 1:
            return Int(logits.argmax(axis=0).item())

        # Precompute 1/temperature once; applied only to survivors (or to all
        # V when truncation is disabled).
        var inv_temp = Scalar[F32](1.0) / Scalar[F32](temperature)

        # Softmax normalizes along the only axis of a 1-D tensor.
        var axes = List[Int]()
        axes.append(0)

        var probs: Tensor[F32]

        if top_k > 1 and top_k < V:
            # Manual selection: repeatedly take the current max and mask it
            # with -inf so the next argmax finds the runner-up. O(top_k * V)
            # total. At k=50, V=256 that is ~12.8k comparisons — negligible.
            # If V or k grows, swap this for a min-heap (O(V log k)) or
            # quickselect (O(V) avg). Selection runs on raw logits (see
            # docstring) and we snapshot the *scaled* value at pick time, so
            # logits is never re-read and no full-V division pass is needed.
            var work = logits.clone()
            var kept = List[Int](capacity=top_k)
            var kept_values = List[Scalar[F32]](capacity=top_k)
            for _ in range(top_k):
                var m = Int(work.argmax(axis=0).item())
                kept.append(m)
                kept_values.append(work[m] * inv_temp)
                work[m] = neg_inf[F32]()

            # Build the masked logit vector: -inf everywhere, scaled survivors
            # in their original positions. exp(-inf) == 0, so softmax gives
            # the masked entries exactly zero probability.
            var masked = Tensor[F32].full(Shape(V), neg_inf[F32]())
            for i in range(top_k):
                masked[kept[i]] = kept_values[i]

            probs = masked.softmax[track_grad=False](axes)
        else:
            # top_k <= 1 is handled above; top_k >= V means "keep everything".
            if temperature != 1.0:
                var scaled = logits.clone() * inv_temp
                probs = scaled.softmax[track_grad=False](axes)
            else:
                # temperature == 1 is a no-op; skip the clone and softmax the
                # input directly.
                probs = logits.softmax[track_grad=False](axes)

        # One sample, so `replacement` is inert here; kept for callers who
        # later bump num_samples. `init_seed == None` → fresh entropy inside
        # `multinomial`; `init_seed != None` → reproducible draw.
        var drawn = probs.multinomial(
            num_samples=1, replacement=False, init_seed=init_seed
        )
        return Int(drawn.item())


struct Loss:
    @staticmethod
    def estimate(
        mut model: CharGPT,
        ref dataset: Dataset,
        eval_iters: Int,
    ) raises -> Dict[String, Float32]:
        """
        Loss.estimate: a steadier loss reading
        The loss of a single random batch is noisy. Averaging `eval_iters` batches
        from each split gives a number that is comparable from one evaluation to the
        next. Watch the two numbers together: train and val falling together means
        the model is learning the language; val stalling or rising while train keeps
        falling means it is starting to memorise the training text.
        """
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
        # hand the model back in training mode
        model.train()
        return out^


def generate_text(
    tokenizer: BPETokenizer[_],
    mut model: CharGPT,
    dataset: Dataset,
    max_new_tokens: Int,
    temperature: Float32 = 0.8,
    top_k: Int = 50,
    init_seed: Optional[Int] = None,
) raises:
    """
    Show what the model has learned so far.
    The starting prompt is 5 consecutive characters from a random place in the
    validation text, which the model never trains on, so the sample is not
    something it could simply be repeating from memory.
    """

    model.eval()
    # Get a random single batch from validation data with 5 char sequence
    var rand_batch = dataset.random_batch("val", 1, 5)  # tuple of tensors
    var rand_context = rand_batch[0]
    var generated = model.generate(
        rand_context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        init_seed=init_seed,
    )
    # Row 0 of the (1, T+max_new_tokens) result, turned back into text.
    var tokens = generated.tolist()
    var text = tokenizer.decode(tokens)
    print(text)
    model.train()


def main() raises:
    # Settings come from key=value arguments; anything not given keeps its
    # default. An unknown key or an unparsable value is silently ignored,
    # so check the "Hyper params" line printed below.
    var h_params = HParams(argv())
    print("Hyper params: ", h_params)
    var text = TinyShakespeare.text()
    # We are training a character model - not a subword model
    # With mbpe's BPETokenizer we should be able to handle any unicode character thrown at it - not necessarily ascii
    var tokenizer = BPETokenizer[CharPretokenizer]()
    var corpus = Span[String]([text])
    # The corpus has 65 unique ascii - as can be found in accompanying bigram model
    # Requesting 300 token(char) would still produce 256 vocab - BPE base starting vocab(255 bytes)
    tokenizer.train(corpus, 300)

    var vocab_size = len(tokenizer)
    # Would print 256
    print("Tokenizer len: ", len(tokenizer))
    # Create tensor from text data
    var data = Tensor[INT].d1(tokenizer.encode(text^))
    # Create the dataset to train and test from
    var dataset = Dataset(data^, h_params)
    # The character based generative model
    var model = CharGPT(vocab_size, h_params)
    # AdamW keeps a running average of each weight's gradient and of its
    # square, and scales each step by them. Gradient clipping is available
    # but switched off here.
    var optimizer = AdamW[F32](
        model.parameters(),
        lr=Scalar[F32](h_params.learning_rate),
        # clip_norm=Scalar[F32](1),
        # clip_value=Scalar[F32](0.5),
    )
    model.train()
    # Before any training: every weight uses Kaiming (He) init, which keeps
    # the initial logits at roughly unit variance. Cross-entropy of random
    # logits is ln(256) ≈ 5.55 plus an overconfidence penalty, so ~6.6 is
    # expected here, not 5.55 (that would need near-zero logits). A value far
    # from ~6.6 would indicate a broken init or forward pass.
    var losses = Loss.estimate(model, dataset, h_params.eval_iters)
    print(
        "Starting: train loss {0}, val loss {1}".format(
            losses["train"], losses["val"]
        )
    )
    print()
    generate_text(
        tokenizer,
        model,
        dataset,
        h_params.max_new_tokens,
    )

    print()
    # The training loop. One step = one batch:
    # forward -> loss -> clear old gradients -> backward -> weight update
    for step in range(h_params.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if (step + 1) % h_params.eval_interval == 0:
            var losses = Loss.estimate(model, dataset, h_params.eval_iters)
            print(
                "Step {0}: train loss {1}, val loss {2}".format(
                    step + 1, losses["train"], losses["val"]
                )
            )
            print()
            generate_text(
                tokenizer,
                model,
                dataset,
                h_params.max_new_tokens,
            )

        var (xb, yb) = dataset.random_batch("train")
        var output = model(xb, yb)
        ref loss = output[1]
        # loss is returned as Optional - hence `value()`
        ref loss_inner = loss.value()
        # gradients accumulate, so clear the last step's
        optimizer.zero_grad()
        # how should each weight change to lower loss?
        loss_inner.backward()
        # nudge every weight a little in that direction
        optimizer.step()

    model.eval()
    # A final sample from the trained model.
    generate_text(
        tokenizer,
        model,
        dataset,
        h_params.max_new_tokens,
    )
