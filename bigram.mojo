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

from shared import CmdLineArgs, TinyShakespeare, learning_rate

comptime INT = DType.int
comptime INT64 = DType.int64
comptime F32 = DType.float32

comptime HParams = CmdLineArgs


struct CharTokenizer(Sized):
    var stoi: Dict[String, Int]
    var itos: Dict[Int, String]

    def __init__(out self, var text: String):
        var byte_set = Set[Byte]([byte for byte in text.bytes()])
        var ints_from_bytes = [Int(byte) for byte in byte_set]
        sort(ints_from_bytes)
        var chars_as_string = [chr(each) for each in ints_from_bytes]
        var vocab = "".join(chars_as_string)
        print("Vocabulary: ", vocab)
        self.stoi = {s: i for i, s in enumerate(chars_as_string)}
        self.itos = {i: s for i, s in enumerate(chars_as_string)}

    def encode(self, s: String) raises -> List[Int]:
        return [self.stoi[chr(Int(byte))] for byte in s.bytes()]

    def decode(self, tokens: List[Int]) raises -> String:
        return "".join([self.itos[token] for token in tokens])

    def __len__(self) -> Int:
        return len(self.itos)


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
        var logits = self.embedding_table(indices.to_dtype[INT64]())
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
        return self.embedding_table.parameters()

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
        # indices (B, T) from current context
        var appended = indices.clone()
        for _ in range(max_new_tokens):
            var (logits, _) = self(appended)
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
        return appended^


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

    def train_ds(ref self) -> ref[self.train_data] Tensor[INT]:
        return self.train_data

    def val_ds(ref self) -> ref[self.val_data] Tensor[INT]:
        return self.val_data

    def get_batch(
        self, mode: String
    ) raises -> Tuple[Tensor[INT], Tensor[INT]]:
        ref data_src = self.train_data if mode == "train" else self.val_data
        var indices = Tensor[INT].rand(
            self.batch_size, high=len(data_src) - self.seq_length
        )
        var x = Tensor[INT].stack(
            [data_src[idx : idx + self.seq_length] for _, idx in indices]
        )
        var y = Tensor[INT].stack(
            [data_src[idx + 1 : idx + self.seq_length + 1] for _, idx in indices]
        )
        return x^, y^


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
                var (x, y) = dataset.get_batch(mode)
                var (_, loss) = model(x, y)
                # loss.value() because loss is Optional
                losses[k] = loss.value().item()
            out[mode] = losses.mean().item()
        model.train()
        return out^


def main() raises:
    var h_params = CmdLineArgs(argv())
    print("Hyper params: ", h_params)
    var text = TinyShakespeare.text()
    var tokenizer = CharTokenizer(text)

    var vocab_size = len(tokenizer)
    print("Tokenizer len: ", len(tokenizer))
    var data = Tensor[INT].d1(tokenizer.encode(text^))
    var dataset = Dataset(data^, h_params)
    var model = BigramModel(vocab_size)

    print(
        tokenizer.decode(
            model.generate(
                Tensor[INT].zeros(1, 1),
                max_new_tokens=h_params.max_new_tokens,
                max_seed=h_params.seed_range,
            )[Indexer(0), Slicer()].tolist()
        )
    )

    var optimizer = AdamW[F32](
        model.parameters(),
        lr=Scalar[F32](learning_rate),
    )

    for iter in range(h_params.max_iters):
        # every once in a while evaluate the loss on train and val sets
        if (iter + 1) % h_params.eval_interval == 0:
            var losses = Loss.estimate(model, dataset, h_params.eval_iters)
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
