# chargpt

A small character-level GPT in [Mojo](https://www.modular.com/mojo), built step by step on CPU. It trains on Tiny Shakespeare and writes Shakespeare-shaped text, one character at a time.

It is built on two libraries of mine:

- [**Tenmo**](https://github.com/ratulb/tenmo): tensors, autograd, layers and the AdamW optimizer.
- [**mbpe**](https://github.com/ratulb/mbpe): the tokenizer. The character-level behaviour comes from a single small class, `CharPretokenizer`, described [below](#how-it-uses-mbpe).

This is a teaching project, not a serious language model. It is deliberately CPU-only and uncluttered, so you can read each file in one sitting. It is the smallest sibling of [*Tokens to Transformers in Mojo*](https://ratulb.github.io/tokens-to-transformers/), the book that builds a GPT-2 from tokenizer to training loop.

## The learning path

The project is meant to be read in order. Each stage adds one idea to the one before.

| Stage | File | What it adds |
| --- | --- | --- |
| 1 | `bigram.mojo` | The baseline: predict the next character from the current one alone, with no context |
| 2 | `chargpt.mojo` | A transformer: position embeddings, causal attention, MLP blocks, residuals and layer norm |

Ideas beyond that, not yet started: a learning-rate schedule and gradient clipping, saving and loading checkpoints with a separate `generate` command, dropout as a command-line argument, and a word-level pre-tokenizer to compare subword tokens against characters. See [Where it goes next](#where-it-goes-next).

## What it looks like

An early run with a context of only 8 characters (validation loss 2.33, about 3.4 bits per character):

```
ht as 'akeat?
And, thoso amy

My ankn ao me uhiemis
h
If Cour wet a
And I is thers hatbut
```

A later run, with a context of 32 and a model of about 0.7 million parameters (validation loss about 1.79, about 2.6 bits per character):

```
d with counst then, let be alont:
Ay, let us, our it.
And and i't my life; mort true agot narther.

KYFING RICHARD II:
No kepper thee. Lyman, mawder, foollows, what un to at lonce, my dotth not parts
At in the brid wants; I breather't them, and would tell; talkive to knot Mishing or prectrues,
```

It still invents words, but look at what it has learned: speaker names in capitals followed by a colon, short lines, punctuation, and a good share of real words. The first sample had spaces and line breaks and little else. That difference is mostly context length and training time. See [Results](#results).

## Quick start

You need:

- [Mojo](https://docs.modular.com/mojo/manual/) 1.1.0 (the version used throughout the book), set up with [pixi](https://pixi.sh).
- `wget`, used once to download the data to `/tmp/input.txt`.

```bash
git clone https://github.com/ratulb/chargpt
cd chargpt
pixi shell
mojo bigram.mojo        # stage 1: the baseline
mojo chargpt.mojo       # stage 2: the transformer, with default settings
```

Change any setting by adding `key=value` arguments (no spaces around the `=`):

```bash
mojo chargpt.mojo batch_size=16 seq_length=32 max_iters=15000 n_embd=96 n_layer=6 n_head=6
```

To keep a long run going if your remote connection drops:

```bash
setsid nohup mojo chargpt.mojo > chargpt_run.log 2>&1 < /dev/null &
echo $! > chargpt_run.pid
```

## Files

| File | What is in it |
| --- | --- |
| `shared.mojo` | `CmdLineArgs` (every setting, with defaults, read from `key=value` arguments) and `TinyShakespeare` (downloads and reads the text). Used by both programs |
| `bigram.mojo` | Stage 1, the baseline |
| `chargpt.mojo` | Stage 2: the tokenizer hook, the data, the model, the training loop and the sampler |

## How a run works

```
input.txt → CharPretokenizer → BPE core (no merges) → ids → 90% train / 10% validation
                                                                      │
                                   random windows of seq_length ids ◄─┘
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

Each training example is a window of `seq_length` characters, and its target is the same window shifted by one character: at every position the model is asked to predict the next character.

## Command-line arguments

Every setting below can be overridden as `key=value`. They are runtime arguments on purpose. A `comptime` constant is baked in when the program compiles, so changing one means a full recompile, which is slow; an argument means you can run experiments back to back.

| Argument | Default | What it does |
| --- | --- | --- |
| `seq_length` | 32 | Context length in characters. The biggest lever on coherence |
| `batch_size` | 32 | Windows per step |
| `max_iters` | 10000 | Training steps |
| `n_embd` | 192 | Width of every vector |
| `n_head` | 6 | Attention heads. `n_embd` must divide evenly: the default gives 32 numbers per head |
| `n_layer` | 6 | Transformer blocks |
| `eval_interval` | 300 | Steps between evaluations and samples |
| `eval_iters` | 50 | Batches averaged per evaluation. Raise it for steadier numbers |
| `max_new_tokens` | 500 | Length of each sample |
| `gen_interval` | 500 | Steps between generated samples |
| `seed_range` | 1000000 | Each sampled character draws its randomness from a seed in `[0, seed_range]`. Keep it large; a small range limits how varied the samples can be |

Two settings are still compile-time constants, so changing them means recompiling: `learning_rate` (1e-3, in `shared.mojo`) and `drop_ratio` (0.0, at the top of `chargpt.mojo`).

Things to know about the arguments:

- A misspelled key is **ignored silently**, and a value that is not a number falls back to the default. Check the `Hyper params:` line the script prints at the start to see what it actually used.
- If the same key appears twice, the first one wins.

With the defaults, each step sees 32 × 32 = 1,024 characters, so 10,000 steps see about 10.2 million characters: roughly 9 passes over the 1.1 million characters of text. The model has about 2.8 million parameters (a hand count; the script does not print it).

## Reading the loss

Losses are in nats per character. Divide by ln 2 ≈ 0.693 to get bits per character, which is easier to feel.

| Loss | Nats | Bits per character |
| --- | --- | --- |
| Knows nothing (uniform over 256 byte values) | 5.55 | 8.0 |
| Knows nothing (uniform over the 65 characters that occur) | 4.17 | 6.0 |
| Early run, context of 8 | 2.33 | 3.4 |
| Later run, context of 32 | 1.79 | 2.6 |
| nanoGPT's character-level Shakespeare config, on a GPU | about 1.47 | about 2.1 |

A bigram model on this text typically lands near 2.5 nats (Karpathy reports about that in his video). The nanoGPT figure uses a 256-character context and far more training. Treat both as reference points, not as targets.

### The starting loss

Before the first step the script prints a starting loss and a sample of pure noise. With the default settings that loss is about 40, far above ln 256 ≈ 5.55, because the weights are initialized from a standard normal with standard deviation 1. That makes the initial logits large, and cross-entropy then penalizes confidently wrong predictions. This is expected: the optimizer pulls the weights down during the first few hundred steps, and by step 300 the loss is already around 3.4. If you want a starting loss near 5.55 you can switch `init_method` from `"normal"` to an Xavier/He initializer; that is not the default here so that the training curves above stay comparable.

## Results

All runs use Tiny Shakespeare, a 90/10 train/validation split, and AdamW at a constant 1e-3.

**Stage 1, the bigram baseline.** Run `mojo bigram.mojo` and record the final train and validation loss here.

<!-- TODO: add the bigram's final train and validation loss. -->

**Stage 2, an early run:** 5 layers, 320 wide, 5 heads, `seq_length` 8, `batch_size` 32, dropout 0.1, 5,000 steps (about 1.15 passes over the data).

| Step | Train loss | Validation loss |
| --- | --- | --- |
| 300 | 3.97 | 3.98 |
| 1,500 | 2.90 | 2.89 |
| 3,000 | 2.53 | 2.54 |
| 4,800 | 2.31 | 2.33 |

Train and validation stayed close and both were still falling: that run was under-trained, not over-fitted.

**Stage 2, a later run:** 6 layers, 96 wide, 6 heads (32 numbers per head), `seq_length` 32, `batch_size` 16, no dropout, 15,000 steps (about 7 passes), roughly 0.7 million parameters.

```bash
mojo chargpt.mojo batch_size=16 seq_length=32 max_iters=15000 n_embd=96 n_layer=6 n_head=6
```

| Step | Train loss | Validation loss |
| --- | --- | --- |
| 13,500 | 1.67 | 1.83 |
| 13,800 | 1.64 | 1.78 |
| 14,100 | 1.66 | 1.79 |

Each reading averages only 50 batches, so differences of a few hundredths between rows are noise. The gap of about 0.13 between train and validation is small but real: the model has begun to fit the training text a little better than the validation text.

## Where the code is

Everything below is in `chargpt.mojo`, in the order it appears. The file is commented so you can read it top to bottom.

| Piece | What it does |
| --- | --- |
| `CharPretokenizer` | The mbpe pre-tokenizer that makes this a character model |
| `Dataset` | The 90/10 split, and `random_batch`: random windows of the text, with targets shifted by one character |
| `MultiHeadAttention` | Fused query/key/value projection, causal mask, scaled dot product, heads, output projection |
| `MLP` | Widen by four, GELU, narrow back, dropout |
| `Block` | One transformer block: two layer norms, attention and the MLP, each added back to its input (a residual) |
| `CharGPT` | The model: embeddings, blocks, final layer norm, the output head, the loss, and `generate` |
| `Loss.estimate` | Average loss over several random batches of train and validation, with dropout off |
| `generate_text` | Takes a random 5-character prompt from the validation text and generates from it |
| `main` | Reads the arguments, builds the tokenizer, data, model and optimizer, then runs the training loop |

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
- **A key-value cache.** Generation recomputes the whole context at each step, which is fine at this size.
- **Weight tying and checkpoint saving.** Training starts from scratch every time, and the trained model is not saved.
- **Configuration files.** Settings are `key=value` arguments; there is no config file.

## Where it goes next

Planned, in order:

1. **Temperature and top-k sampling.** Today `generate` samples from the model's raw distribution. Dividing the logits by a temperature below 1 makes the text more conservative, and keeping only the k most likely characters removes the unlikely tail, so samples contain fewer invented words. The loss does not change; only the sampling does.

Ideas, not yet started:

2. **A better training recipe.** A learning-rate schedule (a short warmup, then decay), gradient clipping, and dropout as a command-line argument.
3. **Checkpoints.** Save the trained weights, and add a separate `generate` command so a sample does not need a retrain.
4. **A word-level pre-tokenizer.** Compare characters against subword tokens on the same model.
5. **A GPU path.** A small, separate file, so the main code stays uncluttered.

## Things to try

Predict first, then run.

1. **Context.** Run with `seq_length=8`, then `32`, then `128`. What happens to the loss and to the text? Context costs little compute at this width.
2. **Overfit one batch.** Train on a single fixed batch. The loss should fall close to zero, which proves the plumbing works.
3. **Width and heads.** Keep `n_embd` fixed and change `n_head`. Fewer, wider heads against more, narrower ones: which does the loss prefer?
4. **Dropout.** Change `drop_ratio` from 0.0 to 0.1 (this needs a recompile) and train for about 9 passes. Does the gap between train and validation loss change?
5. **Position embeddings.** Remove them. Can the model still use word order?
6. **The mask.** Remove the causal mask in `MultiHeadAttention`. Training loss will fall too well, because the model can read the answer; generation will not work. Why?
7. **The pre-tokenizer.** Swap `CharPretokenizer` for a word-splitting one and train to a larger vocabulary.
8. **Sampling.** Add a temperature to `generate` and compare samples from one trained model at 1.0, 0.8 and 0.5.

## Credits

- The Tiny Shakespeare text comes from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repository.
- The structure follows Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and his video [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY).
- [Tenmo](https://github.com/ratulb/tenmo) and [mbpe](https://github.com/ratulb/mbpe) are the libraries underneath; [*Tokens to Transformers in Mojo*](https://ratulb.github.io/tokens-to-transformers/) explains how they work.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
