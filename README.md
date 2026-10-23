# chargpt

A small character-level GPT in [Mojo](https://www.modular.com/mojo), in one file, on CPU. It trains on Tiny Shakespeare and writes Shakespeare-shaped text, one character at a time.

It is built on two libraries of mine:

- [**Tenmo**](https://github.com/ratulb/tenmo): tensors, autograd, layers and the AdamW optimizer.
- [**mbpe**](https://github.com/ratulb/mbpe): the tokenizer. The character-level behaviour comes from a single small class, `CharPretokenizer`, described below.

This is a teaching project, not a serious language model. It is deliberately CPU-only and uncluttered, so you can read the whole thing in one sitting. It is the smallest sibling of [*Tokens to Transformers in Mojo*](https://ratulb.github.io/tokens-to-transformers/), the book that builds a GPT-2 from tokenizer to training loop.

## What it looks like

This is real output from an early run with a context of only 8 characters (validation loss about 2.33):

```
ht as 'akeat?
And, thoso amy

My ankn ao me uhiemis
h
If Cour wet a
And I is thers hatbut
I angreos br To foure!r youl, foure prealy your frisf pll bather wate, of ronce ans for of he's mabo your, byoupe,
```

No real words, but notice what it has learned: spaces, capitals, line breaks, punctuation, and word-shaped fragments. With a longer context and more training the fragments turn into words. See [Results](#results).

## Quick start

You need:

- [Mojo](https://docs.modular.com/mojo/manual/) 1.1.0 (the version used throughout the book).
- `wget`, used once to download the data to `/tmp/input.txt`.

```bash
git clone https://github.com/ratulb/chargpt
cd chargpt
pixi shell
mojo chargpt.mojo
```

To launch traing in detached mode(to survive remote connection break):

```bash
setsid nohup mojo chargpt.mojo > chargpt_run.log 2>&1 < /dev/null & \
echo $! > chargpt_run.pid
```

Every 300 steps the script prints the training and validation loss, then a 500-character sample. The prompt for each sample is five consecutive characters taken from a random place in the validation text, so the model never sees its own prompt during training. A final sample is printed when training ends.

## What happens when you run it

```
input.txt → CharPretokenizer → BPE core (no merges) → ids → 90% train / 10% validation
                                                                      │
                                  random windows of block_size ids ◄──┘
                                                │
                            token + position embeddings
                                                │
                       n_layer × [ LayerNorm → causal attention → add
                                   LayerNorm → MLP (GELU)       → add ]
                                                │
                                   LayerNorm → linear head → logits
                                                │
                          cross-entropy loss → backward → AdamW step
                                                │
                    sampling: softmax → multinomial → append → repeat
```

## Where things are in `chargpt.mojo`

| Piece | What it does |
| --- | --- |
| Constants at the top | All settings, in one place (see below) |
| `TinyShakespeare` | Downloads the text if it is not already in `/tmp` |
| `Dataset` | 90/10 split and `random_batch`: random windows, with targets shifted by one character |
| `Loss.estimate` | Average loss over several random batches of train and validation, with dropout off |
| `CharPretokenizer` | The mbpe pre-tokenizer that makes this a character model |
| `MultiHeadAttention` | Fused QKV projection, causal mask, scaled dot product, heads, output projection |
| `MLP` | Widen by four, GELU, narrow, dropout |
| `Block` | One transformer block: two LayerNorms, attention and MLP, each with a residual |
| `CharGPT` | The model: embeddings, blocks, final LayerNorm, head, loss, and `generate` |
| `generate_text` | Takes a random 5-character prompt from validation and generates from it |
| `main` | Tokenizer, data, model, optimizer, training loop |

## Settings

Everything lives in the constants at the top of the file.

| Constant | Value | What it does |
| --- | --- | --- |
| `block_size` | 48 | Context length in characters. The biggest lever on coherence |
| `batch_size` | 32 | Windows per step |
| `max_iters` | 8000 | Training steps |
| `learning_rate` | 1e-3 | AdamW learning rate, constant (no schedule) |
| `n_embd` | 320 | Width of every vector |
| `n_head` | 5 | Attention heads, so 64 numbers per head |
| `n_layer` | 5 | Transformer blocks |
| `drop_ratio` | 0.1 | Dropout in attention and the MLP |
| `eval_interval` | 300 | Steps between evaluations and samples |
| `eval_iters` | 50 | Batches averaged per evaluation |
| `max_new_tokens` | 500 | Length of each sample |

Each step sees 32 × 48 = 1,536 characters, so 8,000 steps see about 12.3 million characters: roughly 11 passes over the 1.1 million characters of text. The model has about 6.3 million parameters (a hand count; the script does not print it).

## Results

For scale, the loss of a model that knows nothing is ln 256 ≈ 5.55 nats, because the vocabulary is 256 byte values. Divide a loss in nats by ln 2 to get bits per character.

An early run with `block_size = 8`, batch 32, 5,000 steps (about 1.15 passes over the data):

| Step | Train loss | Validation loss |
| --- | --- | --- |
| 300 | 3.97 | 3.98 |
| 1,500 | 2.90 | 2.89 |
| 3,000 | 2.53 | 2.54 |
| 4,800 | 2.31 | 2.33 |

That is about 3.4 bits per character. Train and validation stayed close together and both were still falling, so that run was under-trained rather than over-fitted. For comparison, [nanoGPT](https://github.com/karpathy/nanoGPT)'s character-level Shakespeare config, with a 256-character context and far more tokens per step, reaches a validation loss of about 1.47 (about 2.1 bits).

The current settings (`block_size = 48`, 8,000 steps) have not been written up yet. Run them and add your numbers here:

| Step | Train loss | Validation loss |
| --- | --- | --- |
| 8,000 | | |

## How it uses mbpe

[mbpe](https://github.com/ratulb/mbpe) is a byte-level BPE tokenizer. Its core knows only BPE; everything that makes one tokenizer different from another lives in a *pre-tokenizer*, a small class that cuts text into pieces before any merging. To get a character model, `chargpt.mojo` implements the pre-tokenizer trait and nothing else:

```mojo
var tokenizer = BPETokenizer[CharPretokenizer]()
tokenizer.train(corpus, 300)
```

`CharPretokenizer.split` returns one piece per Unicode code point. BPE only merges bytes *within* a piece, so with every character in its own piece no merge ever happens. That is why the vocabulary stays at the 256 base byte values even though `train` is asked for 300, and why the tokenizer length prints as 256 rather than the 65 distinct characters in the text.

Because the base is bytes, any UTF-8 text round-trips, and a non-ASCII character simply becomes several byte tokens. Nothing is ever out of vocabulary.

The model code does not know any of this. Swap in a pre-tokenizer that splits on words, train to a larger vocabulary, and the same model trains on subword tokens, with shorter sequences. That one-line swap is the quickest way to see what BPE adds.

## What it deliberately leaves out

- **GPU.** Everything runs on CPU, to keep the code small and free of device plumbing.
- **A learning-rate schedule, gradient clipping, weight tying and checkpoint saving.** Training runs from scratch every time.
- **A key-value cache.** Generation recomputes the whole context at each step, which is fine at this size.
- **Configuration files or command-line flags.** The constants at the top are the interface.

## Things to try

Predict first, then run.

1. **Context.** Change `block_size` from 48 to 8, then to 128. What happens to the loss and to the text? Context costs little compute at this width.
2. **Overfit one batch.** Train on a single fixed batch. The loss should fall close to zero, which proves the plumbing works.
3. **Dropout.** Set `drop_ratio` to 0.0. Over 11 passes, does the gap between train and validation loss change?
4. **Position embeddings.** Remove them. Can the model still use word order?
5. **The mask.** Remove the causal mask in `MultiHeadAttention`. Training loss will fall too well, because the model can read the answer; generation will not work. Why?
6. **The pre-tokenizer.** Swap `CharPretokenizer` for a word-splitting one and train to a larger vocabulary.
7. **Sampling.** Add a temperature and a top-k filter to `generate`.
8. **Learning rate.** Add warmup and cosine decay, and compare the final loss.

## Credits

- The Tiny Shakespeare text comes from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repository.
- The structure follows Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his video [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- [Tenmo](https://github.com/ratulb/tenmo) and [mbpe](https://github.com/ratulb/mbpe) are the libraries underneath; [*Tokens to Transformers in Mojo*](https://ratulb.github.io/tokens-to-transformers/) explains how they work.


---

## License

MIT License - see [LICENSE](LICENSE) for details.

---
