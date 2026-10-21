from std.pathlib import Path
from std.os.process import Process
from std.collections import Set
from std.utils.numerics import neg_inf
from std.math import sqrt
from std.testing import assert_true


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

comptime download_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
comptime input_file = "/tmp/input.txt"

comptime block_size = 8
comptime batch_size = 32
comptime eval_iters = 200
comptime max_iters = 10000
comptime eval_interval = 300
comptime gen_interval = 500
comptime learning_rate = 1e-3
comptime n_embd = 32
comptime drop_ratio = 0.2

comptime INT = DType.int
comptime INT64 = DType.int64
comptime F32 = DType.float32


struct TinyShakespeare:
    @staticmethod
    def text() raises -> String:
        var path = Path(input_file)
        if not path.exists() or not path.is_file():
            Self.download()
        var content = path.read_text()
        print("Total characters: ", content.byte_length())
        print("Total code points: ", content.count_codepoints())
        print("Total graphemes: ", content.count_graphemes())
        print()

        return content^

    @staticmethod
    def download() raises:
        var path = Path(input_file)
        if path.exists() and path.is_file():
            return
        print("downloading aclImdb (~1MB)...")
        var args = List[String]()
        args.append("-P")
        args.append("/tmp")
        args.append(download_url)

        var process = Process.run("wget", args^)
        _ = process.wait()
        print("Tiny shakespeare downloaded")


struct CharPretokenizer(PreTokenizer):
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


struct Head:
    """One head of self-attention."""

    var key: Linear[F32]
    var query: Linear[F32]
    var value: Linear[F32]
    var tril: Tensor[F32]
    var dropout: Dropout[F32]
    var inv_scale: Scalar[F32]

    def __init__(out self, head_size: Int = 16):
        self.key = Linear[F32](
            in_features=n_embd,
            out_features=head_size,
            init_method="normal",
            bias=False,
        )
        self.query = Linear[F32](
            in_features=n_embd,
            out_features=head_size,
            init_method="normal",
            bias=False,
        )
        self.value = Linear[F32](
            in_features=n_embd,
            out_features=head_size,
            init_method="normal",
            bias=False,
        )
        self.tril = Tensor[F32].ones(block_size, block_size).tril()
        self.dropout = Dropout[F32](drop_ratio)
        self.inv_scale = Scalar[F32](1.0 / sqrt(Float64(head_size)))

    def __call__(mut self, x: Tensor[F32]) raises -> Tensor[F32]:
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        ref x_shape = x.shape()
        # var B,T,C = x_shape[0], x_shape[1], x_shape[2]
        var T = x_shape[1]
        var k = self.key(x)  # (B,T,hs)
        var q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        var wei = (
            q.matmul(k.transpose(-1, -2)) * self.inv_scale
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, neg_inf[F32]()
        )  # (B, T, T)
        wei = wei.softmax(axes=[-1])  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        var v = self.value(x)  # (B,T,hs)
        var out = wei.matmul(v)  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out^

    def parameters(ref self) -> List[Pointer[Tensor[F32], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[F32], MutAnyOrigin]]()
        params.extend(self.key.parameters())
        params.extend(self.query.parameters())
        params.extend(self.value.parameters())
        return params^

    def train(mut self):
        self.key.training = True
        self.query.training = True
        self.value.training = True
        self.dropout.training = True

    def eval(mut self):
        self.key.training = False
        self.query.training = False
        self.value.training = False
        self.dropout.training = False


struct BigramModel:
    var token_embdding_table: Embedding[F32]
    var position_embdding_table: Embedding[F32]
    var sa_head: Head
    var lm_head: Linear[F32]
    var criterion: CrossEntropyLoss[F32]

    def __init__(out self, vocab_size: Int):
        self.token_embdding_table = Embedding[F32](
            vocab_size, n_embd, init_method="normal"
        )
        self.position_embdding_table = Embedding[F32](
            block_size, n_embd, init_method="normal"
        )
        self.sa_head = Head(n_embd)
        self.lm_head = Linear[F32](
            in_features=n_embd, out_features=vocab_size, init_method="normal"
        )
        self.criterion = CrossEntropyLoss[F32]()

    def __call__(
        mut self,
        indices: Tensor[INT],
        targets: Optional[Tensor[INT]] = None,
    ) raises -> Tuple[Tensor[F32], Optional[Tensor[F32]]]:
        ref indices_shape = indices.shape()
        var token_count = indices_shape[1]
        var token_embddings = self.token_embdding_table(
            indices.to_dtype[INT64]()
        )  # B, T, n_embd
        var positional_embeddings = self.position_embdding_table(
            Tensor[INT64].arange(Int64(token_count))
        )  # T, n_embd
        var combined_embeddings = (
            token_embddings + positional_embeddings
        )  # B, T, n_embd
        var attended_embeddings = self.sa_head(combined_embeddings)
        var logits = self.lm_head(attended_embeddings)  # B, T, vocab_size
        var loss: Optional[Tensor[F32]] = None
        if targets:
            # CrossEntropyLoss logits and target conformation
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
        params.extend(self.sa_head.parameters())
        params.extend(self.lm_head.parameters())
        return params^

    def train(mut self):
        self.token_embdding_table.training = True
        self.position_embdding_table.training = True
        self.sa_head.train()
        self.lm_head.training = True

    def eval(mut self):
        self.token_embdding_table.training = False
        self.position_embdding_table.training = False
        self.sa_head.eval()
        self.lm_head.training = False

    def generate(
        mut self,
        indices: Tensor[INT],
        max_new_tokens: Int,
        max_seed: UInt64 = 100,
    ) raises -> Tensor[INT]:
        # indices (B, T) from current context
        var appended = indices.clone()
        for _ in range(max_new_tokens):
            # Position embeddings only have `block_size` rows, so the model
            # cannot accept a context longer than block_size. Crop the growing
            # generated sequence to its last block_size tokens (like
            # idx[:, -block_size:] in nanoGPT) before each forward pass.
            var T = appended.shape()[1]
            var idx_cond: Tensor[INT]
            if T <= block_size:
                idx_cond = appended.clone()
            else:
                var start = T - block_size
                idx_cond = appended[Slicer(), Slicer(start, None, None)]
            var (logits, _) = self(idx_cond)
            # logits now (B, C)
            logits = logits[Slicer(), -1, Slicer()]
            var probas = logits.softmax[track_grad=False](axes=[-1])
            # next_id's shape (B, 1)
            seed()
            var next_id = probas.multinomial[index_dtype=INT](
                init_seed=Int(random_ui64(0, max_seed))
            )
            appended = Tensor[INT].concat[track_grad=False](
                [appended, next_id], axis=1
            )
        return appended^  # (B, T+1)


struct Dataset:
    var train_data: Tensor[INT]
    var val_data: Tensor[INT]

    def __init__(out self, var data: Tensor[INT], split_ratio: Float64 = 0.9):
        var split = Int(split_ratio * Float64(len(data)))
        self.train_data = data[:split]
        self.val_data = data[split:]

    def train_ds(ref self) -> ref[self.train_data] Tensor[INT]:
        return self.train_data

    def val_ds(ref self) -> ref[self.val_data] Tensor[INT]:
        return self.val_data

    def get_batch(self, mode: String) raises -> Tuple[Tensor[INT], Tensor[INT]]:
        ref data_src = self.train_data if mode == "train" else self.val_data
        var indices = Tensor[INT].rand(
            batch_size, high=len(data_src) - block_size
        )
        var x = Tensor[INT].stack(
            [data_src[i[1] : i[1] + block_size] for i in indices]
        )
        var y = Tensor[INT].stack(
            [data_src[i[1] + 1 : i[1] + block_size + 1] for i in indices]
        )
        return x^, y^


struct Loss:
    @staticmethod
    def estimate(
        mut model: BigramModel,
        ref dataset: Dataset,
    ) raises -> Dict[String, Float32]:
        model.eval()
        var out = {"train": Float32(0.0), "val": Float32(0.0)}
        for mode in ["train", "val"]:
            var losses = Tensor[F32].zeros(eval_iters)
            for k in range(eval_iters):
                var (x, y) = dataset.get_batch(mode)
                var (_, loss) = model(x, y)
                # loss.value() because loss is Optional
                losses[k] = loss.value().item()
            out[mode] = losses.mean().item()
        model.train()
        return out^


def main() raises:
    var text = TinyShakespeare.text()
    var tokenizer = BPETokenizer[CharPretokenizer]()
    var corpus = Span[String]([text])
    tokenizer.train(corpus, 300)

    var vocab_size = len(tokenizer)
    print("Tokenizer len: ", len(tokenizer))
    var data = Tensor[INT].d1(tokenizer.encode(text^))
    var dataset = Dataset(data^)
    var model = BigramModel(vocab_size)
    var optimizer = AdamW[F32](
        model.parameters(),
        lr=Scalar[F32](learning_rate),
        clip_norm=Scalar[F32](1),
        clip_value=Scalar[F32](0.5),
    )

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if (iter + 1) % eval_interval == 0:
            var losses = Loss.estimate(model, dataset)
            print(
                "Step {0}: train loss {1}, val loss {2}".format(
                    iter + 1, losses["train"], losses["val"]
                )
            )

        var (xb, yb) = dataset.get_batch("train")
        var output = model(xb, yb)
        ref _logits = output[0]
        ref loss = output[1]
        # loss is returned as Optional - hence `value()`
        ref loss_inner = loss.value()
        optimizer.zero_grad()
        loss_inner.backward()
        optimizer.step()

    model.eval()
    print(
        tokenizer.decode(
            model.generate(
                Tensor[INT].zeros(1, 1),
                max_new_tokens=500,
                max_seed=UInt64(100),
            )[Indexer(0), Slicer()].tolist()
        ),
        end="\n",
    )
