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
from std.random import random_ui64, seed
from tenmo.adamw import AdamW
from tenmo.dropout import Dropout
from tenmo.net import GeLU
from tenmo.layernorm import LayerNorm
from shared import CmdLineArgs, TinyShakespeare, learning_rate

comptime drop_ratio = 0.0

comptime INT = DType.int
comptime INT64 = DType.int64
comptime F32 = DType.float32

comptime HParams = CmdLineArgs


struct CharPretokenizer(PreTokenizer):
    """Implment `PreTokenizer` trait from mbpe - to be used in conjunction with BPETokenizer.    Text would be token per unicode character codepoint.
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
            # One "word" per codepoint. Clamp so a truncated trailing
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
        var split = Int(split_ratio * Float64(len(data)))
        self.train_data = data[:split]
        self.val_data = data[split:]
        self.batch_size = h_params.batch_size
        self.seq_length = h_params.seq_length

    def random_batch(
        self,
        mode: String,
        batch_size_override: Optional[
            Int
        ] = None,  # Override batch size if need be. For example: Get initial prompt
        seq_length_override: Optional[
            Int
        ] = None,  # Override block size if needed. For example: For getting initail prompt
    ) raises -> Tuple[Tensor[INT], Tensor[INT]]:
        ref data_src = self.train_data if mode == "train" else self.val_data
        var size_of_batch = batch_size_override.or_else(self.batch_size)
        var size_of_block = seq_length_override.or_else(self.seq_length)
        var indices = Tensor[INT].rand(
            size_of_batch, high=len(data_src) - size_of_block
        )
        # In tenmo, iterating over a Tensor produces coord(s) and value(s).
        # We are interested in the values.
        var x = Tensor[INT].stack(
            [data_src[idx : idx + size_of_block] for _, idx in indices]
        )
        var y = Tensor[INT].stack(
            [data_src[idx + 1 : idx + size_of_block + 1] for _, idx in indices]
        )
        return x^, y^


# Multi-head causal self-attention: the decoder's "look back" block.
# Q, K, V come from one fused linear (c_attn); the lower-triangular mask
# (tril) hides future positions so token i attends only to tokens <= i.
struct MultiHeadAttention:
    """Implement multi-head causal attention."""
    var n_head: Int
    var head_dim: Int
    var c_attn: Linear[F32]  # Fused QKV: n_embd -> 3*n_embd
    var c_proj: Linear[F32]
    var tril: Tensor[F32]
    var dropout: Dropout[F32]
    var inv_scale: Scalar[F32]  # Pre-calculated for scaling QK^
    def __init__(out self, n_embd: Int, n_head: Int, seq_length: Int):
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = Linear[F32](
            in_features=n_embd,
            out_features=3 * n_embd,
            init_method="normal",
            bias=False,
        )
        self.c_proj = Linear[F32](
            in_features=n_embd, out_features=n_embd, init_method="normal"
        )
        self.tril = Tensor[F32].ones(seq_length, seq_length).tril()
        self.dropout = Dropout[F32](drop_ratio)
        self.inv_scale = Scalar[F32](1.0 / sqrt(Float64(self.head_dim)))

    def __call__(mut self, x: Tensor[F32]) raises -> Tensor[F32]:
        # Causal multi-head attention forward.
        ref x_shape = x.shape()
        var B = x_shape[0]
        var T = x_shape[1]
        var C = x_shape[2]

        # Fused QKV projection, then split along feature axis.
        var qkv = self.c_attn(x)  # (B, T, 3*C)
        var q = qkv.slice(0, C, axis=2)  # (B, T, C)
        var k = qkv.slice(C, 2 * C, axis=2)  # (B, T, C)
        var v = qkv.slice(2 * C, 3 * C, axis=2)  # (B, T, C)

        q = q.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 1, 3])
        # Alter head_dim and token permutation in one go to avoid transpose later
        k = k.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 3, 1])
        v = v.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 1, 3])
        # No need for transposing k, see comment above
        var scaled_attn_scores = (
            q.matmul(k) * self.inv_scale
        )  # (B, n_head, T, T)
        var masked_scores = scaled_attn_scores.masked_fill(
            self.tril[:T, :T] == 0, neg_inf[F32]()
        )
        var attn_weights = masked_scores.softmax(axes=[-1])
        var attenuated_attn = self.dropout(attn_weights)
        var weighted_values = attenuated_attn.matmul(
            v
        )  # (B, n_head, T, head_dim)

        var values = weighted_values.permute([0, 2, 1, 3]).reshape(B, T, C)
        return self.c_proj(values)

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[F32], MutAnyOrigin]]()
        params.extend(self.c_attn.parameters())
        params.extend(self.c_proj.parameters())
        return params^

    def train(mut self):
        self.c_attn.training = True
        self.c_proj.training = True
        self.dropout.training = True

    def eval(mut self):
        self.c_attn.training = False
        self.c_proj.training = False
        self.dropout.training = False


# Position-wise feed-forward network: widen 4x, GELU, project back, dropout.
struct MLP:
    var c_fc: Linear[F32]
    var gelu: GeLU[F32]
    var c_proj: Linear[F32]
    var dropout: Dropout[F32]
    def __init__(out self, n_embd: Int):
        self.c_fc = Linear[F32](
            in_features=n_embd, out_features=4 * n_embd, init_method="normal"
        )
        self.gelu = GeLU[F32]()
        self.c_proj = Linear[F32](
            in_features=4 * n_embd, out_features=n_embd, init_method="normal"
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


# One pre-norm transformer block: LayerNorm -> causal attention -> residual,
# then LayerNorm -> MLP -> residual.
struct Block:
    var ln1: LayerNorm[F32]
    var attn: MultiHeadAttention
    var ln2: LayerNorm[F32]
    var mlp: MLP
    def __init__(out self, h_params: HParams):
        self.ln1 = LayerNorm[F32](h_params.n_embd)
        self.attn = MultiHeadAttention(
            h_params.n_embd, h_params.n_head, h_params.seq_length
        )
        self.ln2 = LayerNorm[F32](h_params.n_embd)
        self.mlp = MLP(h_params.n_embd)

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


# Decoder-only GPT: token+position embeddings -> N blocks -> final LayerNorm -> lm_head.
struct CharGPT:
    """The character based generative model."""
    var token_embedding_table: Embedding[F32]
    var position_embedding_table: Embedding[F32]
    var blocks: List[Block]
    var ln_f: LayerNorm[F32]
    var lm_head: Linear[F32]
    var criterion: CrossEntropyLoss[F32]
    var h_params: HParams

    def __init__(out self, vocab_size: Int, h_params: HParams):
        self.token_embedding_table = Embedding[F32](
            vocab_size, h_params.n_embd, init_method="normal"
        )
        self.position_embedding_table = Embedding[F32](
            h_params.seq_length, h_params.n_embd, init_method="normal"
        )
        self.blocks = List[Block](capacity=h_params.n_layer)
        for _ in range(h_params.n_layer):
            self.blocks.append(Block(h_params))
        self.ln_f = LayerNorm[F32](h_params.n_embd)
        self.lm_head = Linear[F32](
            in_features=h_params.n_embd,
            out_features=vocab_size,
            init_method="normal",
        )
        self.criterion = CrossEntropyLoss[F32]()
        self.h_params = h_params

    def __call__(
        mut self, indices: Tensor[INT], targets: Optional[Tensor[INT]] = None
    ) raises -> Tuple[Tensor[F32], Optional[Tensor[F32]]]:
        # Forward pass of the decoder: embeddings -> blocks -> head.
        ref indices_shape = indices.shape()
        var token_count = indices_shape[1]

        # Look up token and absolute position embeddings, then add (broadcasts).
        var token_embeddings = self.token_embedding_table(
            indices.to_dtype[INT64]()
        )  # (B, T, n_embd)
        var position_embeddings = self.position_embedding_table(
            Tensor[INT64].arange(Int64(token_count))
        )  # (T, n_embd) broadcasts to (B, T, n_embd)
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
        if targets:
            ref logits_shape = logits.shape()
            var B, T, C = logits_shape[0], logits_shape[1], logits_shape[2]
            loss = self.criterion(
                logits.reshape(B * T, C),
                targets.value().to_dtype[INT64]().reshape(B * T),
            )

        return logits^, loss^

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
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
        seed_range: UInt64,  # multinomial sets seed at `42` if not provided!
    ) raises -> Tensor[INT]:
        # Autoregressive decoding: append one sampled token at a time.
        self.eval()
        # indices (B, T) from current context
        var appended = indices.clone()
        seed()
        for _ in range(max_new_tokens):
            # Position embeddings only have `seq_length` rows, so the model
            # cannot accept a context longer than seq_length. Crop the growing
            # generated sequence to its last seq_length tokens (like
            # idx[:, -seq_length:] in nanoGPT) before each forward pass.
            var T = appended.shape()[1]
            var idx_cond: Tensor[INT]
            if T <= self.h_params.seq_length:
                idx_cond = appended.clone()
            else:
                var start = T - self.h_params.seq_length
                idx_cond = appended[Slicer(), Slicer(start, None, None)]
            var (logits, _) = self(idx_cond)
            # Take logits for the last position -> (B, vocab_size).
            logits = logits[Slicer(), -1, Slicer()]
            # Convert to probabilities and sample next token id.
            var probas = logits.softmax[track_grad=False](axes=[-1])
            # next_id's shape (B, 1)
            var next_id = probas.multinomial[index_dtype=INT](
                init_seed=Int(random_ui64(0, seed_range))
            )
            # Append sampled token to the running sequence.
            appended = Tensor[INT].concat[track_grad=False](
                [appended, next_id], axis=1
            )
        self.train()
        return appended^  # (B, T+max_new_tokens)


struct Loss:
    @staticmethod
    def estimate(
        mut model: CharGPT,
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


def generate_text(
    tokenizer: BPETokenizer[_],
    mut model: CharGPT,
    dataset: Dataset,
    seed_range: UInt64,
    max_new_tokens: Int,
) raises:
    model.eval()
    # Get a random single batch from validation data with 5 char
    var rand_batch = dataset.random_batch("val", 1, 5)
    var rand_context = rand_batch[0]
    var generated = model.generate(
        rand_context, max_new_tokens=max_new_tokens, seed_range=seed_range
    )
    var tokens = generated[Indexer(0), Slicer()].tolist()
    var text = tokenizer.decode(tokens)
    print(text)
    model.train()


def main() raises:
    var h_params = CmdLineArgs(argv())
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

    var optimizer = AdamW[F32](
        model.parameters(),
        lr=Scalar[F32](learning_rate),
        # clip_norm=Scalar[F32](1),
        # clip_value=Scalar[F32](0.5),
    )
    model.train()

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
        h_params.seed_range,
        h_params.max_new_tokens,
    )

    print()

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
                h_params.seed_range,
                h_params.max_new_tokens,
            )

        var (xb, yb) = dataset.random_batch("train")
        var output = model(xb, yb)
        ref loss = output[1]
        # loss is returned as Optional - hence `value()`
        ref loss_inner = loss.value()
        optimizer.zero_grad()
        loss_inner.backward()
        optimizer.step()

    model.eval()
    generate_text(
        tokenizer, model, dataset, h_params.seed_range, h_params.max_new_tokens
    )
