# Char-level GPT in `chargpt.mojo` — build guide

This guide turns the current single-head attention model into a small causal
transformer for character-level text. It keeps the existing data pipeline,
training loop, optimizer, and cropped generation. The attention block uses a
fused QKV projection (`c_attn`) like the tenmo `SelfAttention` design, but
implemented directly in `chargpt.mojo`.

## Target architecture

```text
token_emb + pos_emb
  -> n_layer x Block
       -> LayerNorm
       -> MultiHeadAtten (causal, fused QKV)
       -> residual add
       -> LayerNorm
       -> MLP (4*n_embd -> GELU -> n_embd)
       -> residual add
  -> final LayerNorm
  -> lm_head -> logits
```

## 1. Config and imports

```mojo
from tenmo.net import Linear, GeLU
from tenmo.layernorm import LayerNorm

comptime n_head = 4      # n_embd=32 -> head_dim=8
comptime n_layer = 2
```

## 2. `MultiHeadAtten` (fused QKV)

Replace the old `Head` struct with this.

```mojo
struct MultiHeadAtten(ImplicitlyCopyable):
    var n_head: Int
    var head_dim: Int
    var c_attn: Linear[F32]   # fused QKV: n_embd -> 3*n_embd
    var c_proj: Linear[F32]
    var tril: Tensor[F32]
    var dropout: Dropout[F32]
    var inv_scale: Scalar[F32]

    def __init__(out self, n_embd: Int, n_head: Int):
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = Linear[F32](
            in_features=n_embd,
            out_features=3 * n_embd,
            init_method="normal",
            bias=False,
        )
        self.c_proj = Linear[F32](
            in_features=n_embd,
            out_features=n_embd,
            init_method="normal",
        )
        self.tril = Tensor[F32].ones(block_size, block_size).tril()
        self.dropout = Dropout[F32](drop_ratio)
        self.inv_scale = Scalar[F32](1.0 / sqrt(Float64(self.head_dim)))

    def __call__(mut self, x: Tensor[F32]) raises -> Tensor[F32]:
        ref x_shape = x.shape()
        var B = x_shape[0]
        var T = x_shape[1]
        var C = x_shape[2]

        var qkv = self.c_attn(x)                 # (B, T, 3*C)
        var q = qkv.slice(0, C, axis=2)          # (B, T, C)
        var k = qkv.slice(C, 2 * C, axis=2)      # (B, T, C)
        var v = qkv.slice(2 * C, 3 * C, axis=2) # (B, T, C)

        q = q.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 1, 3])
        k = k.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 1, 3])
        v = v.reshape(B, T, self.n_head, self.head_dim).permute([0, 2, 1, 3])

        var wei = q.matmul(k.permute([0, 1, 3, 2])) * self.inv_scale  # (B, h, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, neg_inf[F32]())
        wei = wei.softmax(axes=[-1])
        wei = self.dropout(wei)

        var out = wei.matmul(v)                                    # (B, h, T, dh)
        out = out.permute([0, 2, 1, 3]).reshape(B, T, C)           # (B, T, C)
        return self.c_proj(out)

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
```

## 3. MLP

```mojo
struct MLP(ImplicitlyCopyable):
    var c_fc: Linear[F32]
    var gelu: GeLU[F32]
    var c_proj: Linear[F32]
    var drop: Dropout[F32]

    def __init__(out self, n_embd: Int):
        self.c_fc = Linear[F32](
            in_features=n_embd,
            out_features=4 * n_embd,
            init_method="normal",
        )
        self.gelu = GeLU[F32]()
        self.c_proj = Linear[F32](
            in_features=4 * n_embd,
            out_features=n_embd,
            init_method="normal",
        )
        self.drop = Dropout[F32](drop_ratio)

    def __call__(mut self, x: Tensor[F32]) -> Tensor[F32]:
        var h = self.c_fc(x)
        h = self.gelu(h)
        h = self.c_proj(h)
        return self.drop(h)

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[F32], MutAnyOrigin]]()
        params.extend(self.c_fc.parameters())
        params.extend(self.c_proj.parameters())
        return params^

    def train(mut self):
        self.c_fc.training = True
        self.gelu.train()
        self.c_proj.training = True
        self.drop.training = True

    def eval(mut self):
        self.c_fc.training = False
        self.gelu.eval()
        self.c_proj.training = False
        self.drop.training = False
```

## 4. Transformer block (pre-norm + residuals)

```mojo
struct Block(ImplicitlyCopyable):
    var ln1: LayerNorm[F32]
    var attn: MultiHeadAtten
    var ln2: LayerNorm[F32]
    var mlp: MLP

    def __init__(out self, n_embd: Int, n_head: Int):
        self.ln1 = LayerNorm[F32](n_embd)
        self.attn = MultiHeadAtten(n_embd, n_head)
        self.ln2 = LayerNorm[F32](n_embd)
        self.mlp = MLP(n_embd)

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
```

## 5. Update `BigramModel`

```mojo
struct BigramModel:
    var token_embdding_table: Embedding[F32]
    var position_embdding_table: Embedding[F32]
    var blocks: List[Block]
    var ln_f: LayerNorm[F32]
    var lm_head: Linear[F32]
    var criterion: CrossEntropyLoss[F32]

    def __init__(out self, vocab_size: Int):
        self.token_embdding_table = Embedding[F32](
            vocab_size, n_embd, init_method="normal"
        )
        self.position_embdding_table = Embedding[F32](
            block_size, n_embd, init_method="normal"
        )
        self.blocks = List[Block]()
        for _ in range(n_layer):
            self.blocks.append(Block(n_embd, n_head))
        self.ln_f = LayerNorm[F32](n_embd)
        self.lm_head = Linear[F32](
            in_features=n_embd,
            out_features=vocab_size,
            init_method="normal",
        )
        self.criterion = CrossEntropyLoss[F32]()

    def __call__(
        mut self,
        indices: Tensor[INT],
        targets: Optional[Tensor[INT]] = None,
    ) raises -> Tuple[Tensor[F32], Optional[Tensor[F32]]]:
        ref indices_shape = indices.shape()
        var token_count = indices_shape[1]

        var token_embddings = self.token_embdding_table(indices.to_dtype[INT64]())
        var positional_embeddings = self.position_embdding_table(
            Tensor[INT64].arange(Int64(token_count))
        )
        var combined_embeddings = token_embddings + positional_embeddings

        var x = combined_embeddings
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            x = block(x)
        x = self.ln_f(x)

        var logits = self.lm_head(x)
        var loss: Optional[Tensor[F32]] = None
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
        params.extend(self.token_embdding_table.parameters())
        params.extend(self.position_embdding_table.parameters())
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            ref bp = block.parameters()
            for p in range(len(bp)):
                params.append(bp[p])
        params.extend(self.ln_f.parameters())
        params.extend(self.lm_head.parameters())
        return params^

    def train(mut self):
        self.token_embdding_table.training = True
        self.position_embdding_table.training = True
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            block.train()
        self.ln_f.train()
        self.lm_head.training = True

    def eval(mut self):
        self.token_embdding_table.training = False
        self.position_embdding_table.training = False
        for i in range(len(self.blocks)):
            ref block = self.blocks[i]
            block.eval()
        self.ln_f.eval()
        self.lm_head.training = False
```

## 6. Generation (keep the context crop)

The position table and attention mask both assume at most `block_size` tokens.

```mojo
var T = appended.shape()[1]
var idx_cond: Tensor[INT]
if T <= block_size:
    idx_cond = appended.clone()
else:
    var start = T - block_size
    idx_cond = appended[Slicer(), Slicer(start, None, None)]
var (logits, _) = self(idx_cond)
```

Keep `model.eval()` before generation.

## 7. Validate safely

Use one Mojo process at a time.

1. Temporarily set:
```mojo
comptime max_iters = 10
```
and:
```mojo
max_new_tokens=20
```
2. Run:
```bash
mojo chargpt.mojo
```
3. Confirm it compiles, trains, and prints generated text without a gather panic.
4. Restore:
```mojo
comptime max_iters = 10000
```
and:
```mojo
max_new_tokens=500
```

## Notes

- `n_layer = 2` is a good MVP size for this tiny character-level dataset.
- `drop_ratio = 0.2` may be aggressive; `0.1` can train faster.
- The final `LayerNorm` matters because `lm_head` is untied.
- The fused `c_attn` projection is the efficient GPT-2-style QKV path. If you want the even smaller learning version, split it into three separate `Linear` projections.
- `masked_fill(self.tril[:T, :T] == 0, ...)` keeps the model causal. Because generation crops to `block_size`, `T <= block_size` always holds during inference.
