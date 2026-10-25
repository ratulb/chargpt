# chargpt

A small character-level GPT in [Mojo](https://www.modular.com/mojo), built step by step on CPU. It trains on Tiny Shakespeare and writes Shakespeare-shaped text, one character at a time.

It is built on two libraries:

- [**Tenmo**](https://github.com/ratulb/tenmo): tensors, autograd, layers and the AdamW optimizer.
- [**mbpe**](https://github.com/ratulb/mbpe): the tokenizer. The character-level behaviour comes from a single small class, `CharPretokenizer`, described [below](#how-it-uses-mbpe).

This is a teaching project, not a serious language model. It is deliberately CPU-only and uncluttered, so you can read each file in one sitting. It is the smallest sibling of [*Tokens to Transformers in Mojo*](https://ratulb.github.io/tokens-to-transformers/), the book that builds a GPT-2 from tokenizer to training loop.

## The learning path

The project is meant to be read in order. Each stage adds one idea to the one before.

| Stage | File | What it adds |
| --- | --- | --- |
| 1 | `bigram.mojo` | The baseline: predict the next character from the current one alone, with no context |
| 2 | `chargpt.mojo` | A transformer: position embeddings, causal attention, MLP blocks, residuals and layer norm |

Ideas beyond that, not yet started: a learning-rate schedule and gradient clipping, saving and loading checkpoints with a separate `generate` command, and a word-level pre-tokenizer to compare subword tokens against characters. (Dropout is already a command-line argument via `drop_ratio`.) See [Where it goes next](#where-it-goes-next).

## What it looks like

The model's output changes character as training proceeds, and the *order* in which capabilities appear is the most interesting thing this project shows. Here is the arc, from a 5,000-step run with `seq_length=32`:

**Step 300, validation loss 2.51.** The layout is right — speaker names, colons, line breaks — but the words are noise. The model has learned the cheapest structure first: punctuation and dialogue formatting.

shalll the t be'ce t yo koute.
PANit hindileed youpr, bait moul furereem:

text

**Step 900, validation loss 2.16.** Real words appear (`thou`, `hath`, `doth`) in plausible positions, but no clause structure. Vocabulary emerges before grammar.
Far wall all that lay tre wencrt ith dar a the that gemy you corotanjoy,
Thou all then me lassingeds to thare of a therey astand me arredwaied

text

**Step 1800, validation loss 1.94.** Readable phrases inside incoherent scenes. Clause-level grammar emerges before scene-level coherence.
You would thou come hy wilf swork,
If hown himpay by hader prieny fast gosed
That sacould luked bre me the reblents

text

**Step 3000, validation loss 1.82.** Individual lines are readable. The model has the grammar of a sentence, not the logic of a conversation.
Why, now brother with this pocter:
And me and preaced.

text

**Step 4800, validation loss 1.75.** The plateau. Val stops falling; the model has hit its capacity/context limit. Training longer would overfit (train falls, val rises).
I have should be stand that madam, of not shall he
I'ld, and thing it would as a ganspingnessom'd--it is the most with the king off his for with

text

It still invents words at every stage. But the progression from *layout without vocabulary* to *vocabulary without grammar* to *grammar without coherence* is the same progression a language model of any size goes through. It is worth seeing at this scale, where you can read every file.

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
Change any setting by adding key=value arguments (no spaces around the =):

bash
mojo chargpt.mojo batch_size=16 seq_length=32 max_iters=5000 n_embd=192 n_layer=6 n_head=6
To keep a long run going if your remote connection drops:

bash
setsid nohup mojo chargpt.mojo > chargpt_run.log 2>&1 < /dev/null & echo $! > chargpt_run.pid
Files
File	What is in it
shared.mojo	CmdLineArgs (every setting, with defaults, read from key=value arguments) and TinyShakespeare (downloads and reads the text). Used by both programs
bigram.mojo	Stage 1, the baseline
chargpt.mojo	Stage 2: the tokenizer hook, the data, the model, the training loop and the sampler
kvcache.mojo	The key-value cache: per-layer K/V storage with append, read and sliding-window eviction. Used by the cached generation path
How a run works
text
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
Each training example is a window of seq_length characters, and its target is the same window shifted by one character: at every position the model is asked to predict the next character.

Command-line arguments
Every setting below can be overridden as key=value. They are runtime arguments on purpose. A comptime constant is baked in when the program compiles, so changing one means a full recompile, which is slow; an argument means you can run experiments back to back.

Argument	Default	What it does
seq_length	32	Context length in characters. The biggest lever on coherence
batch_size	32	Windows per step
max_iters	5000	Training steps
n_embd	192	Width of every vector
n_head	6	Attention heads. n_embd must divide evenly: the default gives 32 numbers per head
n_layer	6	Transformer blocks
eval_interval	300	Steps between evaluations and samples
eval_iters	50	Batches averaged per evaluation. Raise it for steadier numbers
max_new_tokens	500	Length of each sample
learning_rate	1e-3	AdamW step size
drop_ratio	0.1	Dropout probability (0.0 = off)
All of these are runtime arguments: changing any of them is just another key=value on the command line, with no recompile. (Nothing here is a compile-time constant anymore.)

Things to know about the arguments:

A misspelled key is ignored silently, and a value that is not a number falls back to the default. Check the Hyper params: line the script prints at the start to see what it actually used.

If the same key appears twice, the first one wins.

With the defaults, each step sees 32 × 32 = 1,024 characters, so 5,000 steps see about 5.1 million characters: roughly 4.6 passes over the 1.1 million characters of text. The model has about 2.8 million parameters (a hand count; the script does not print it).

Reading the loss
Losses are in nats per character. Divide by ln 2 ≈ 0.693 to get bits per character, which is easier to feel.

Loss	Nats	Bits per character
Knows nothing (uniform over 256 byte values)	5.55	8.0
Knows nothing (uniform over the 65 characters that occur)	4.17	6.0
Early run, context of 8	2.33	3.4
Later run, context of 32	1.75	2.5
nanoGPT's character-level Shakespeare config, on a GPU	about 1.47	about 2.1
A bigram model on this text typically lands near 2.5 nats (Karpathy reports about that in his video). The nanoGPT figure uses a 256-character context and far more training. Treat both as reference points, not as targets.

The starting loss
Before the first step the script prints a starting loss and a sample of pure noise. With the default settings that loss is about 6.55 (measured: 6.55 train, 6.55 val), above ln 256 ≈ 5.55, because every weight uses Kaiming (He) initialization, which keeps the initial logits at roughly unit variance. Cross-entropy of random logits is ln V plus an overconfidence penalty that grows with that variance; ln 256 ≈ 5.55 would require near-zero logits. This is expected and healthy: the optimizer pulls the weights down during the first few hundred steps, and by step 300 the loss is already around 2.51 (train) / 2.51 (val). If you want a starting loss near 5.55 you can switch init_method from "kaiming" to a much smaller (or zero) initializer; that is not the default here so that the training curves above stay comparable.

Results
All runs use Tiny Shakespeare, a 90/10 train/validation split, and AdamW at a constant 1e-3.

Stage 1, the bigram baseline. Run mojo bigram.mojo and record the final train and validation loss here.

<!-- TODO: add the bigram's final train and validation loss. -->
Stage 2, an early run: 5 layers, 320 wide, 5 heads, seq_length 8, batch_size 32, dropout 0.1, 5,000 steps (about 1.15 passes over the data).

Step	Train loss	Validation loss
300	3.97	3.98
1,500	2.90	2.89
3,000	2.53	2.54
4,800	2.31	2.33
Train and validation stayed close and both were still falling: that run was under-trained, not over-fitted.

Stage 2, the default run: 6 layers, 192 wide, 6 heads (32 numbers per head), seq_length 32, batch_size 16, dropout 0.1, 5,000 steps (about 4.6 passes), roughly 2.8 million parameters.

bash
mojo chargpt.mojo
Step	Train loss	Validation loss
300	2.51	2.51
900	2.12	2.16
1,500	1.89	2.00
2,100	1.77	1.92
2,700	1.69	1.84
3,300	1.67	1.81
3,900	1.60	1.79
4,500	1.57	1.75
4,800	1.58	1.75
Each reading averages only 50 batches, so differences of a few hundredths between rows are noise. The gap of about 0.17 between train and validation at the end is small but real: the model has begun to fit the training text a little better than the validation text. The validation loss flattens at ~1.75 after step 3,900; this is the model hitting its capacity and context limit, not a training bug.

https://loss_curve_log.svg/

The log-scale plot keeps the initial drop and the later gap in the same picture.

https://loss_curve.svg/

The zoomed plot focuses on the region where train and validation are close, so the small but real gap is visible.

Where the code is
Everything below is in chargpt.mojo, in the order it appears. The file is commented so you can read it top to bottom.

Piece	What it does
CharPretokenizer	The mbpe pre-tokenizer that makes this a character model
Dataset	The 90/10 split, and random_batch: random windows of the text, with targets shifted by one character
MultiHeadAttention	Fused query/key/value projection, causal mask, scaled dot product, heads, output projection. Two __call__ overloads: one uncached (training), one with a KVCache (generation)
MLP	Widen by four, GELU, narrow back, dropout
Block	One transformer block: two layer norms, attention and the MLP, each added back to its input (a residual)
CharGPT	The model: embeddings, blocks, final layer norm, the output head, the loss, and generate
Loss.estimate	Average loss over several random batches of train and validation, with dropout off
generate_text	Takes a random 5-character prompt from the validation text and generates from it
main	Reads the arguments, builds the tokenizer, data, model and optimizer, then runs the training loop
kvcache.mojo is separate and self-contained, with its own test suite in tests/test_kvcache.mojo. See The key-value cache.

How it uses mbpe
mbpe is a byte-level BPE tokenizer. Its core knows only BPE; everything that makes one tokenizer different from another lives in a pre-tokenizer, a small class that cuts text into pieces before any merging. To get a character model, chargpt.mojo implements the pre-tokenizer trait and nothing else:

mojo
var tokenizer = BPETokenizer[CharPretokenizer]()
tokenizer.train(corpus, 300)
CharPretokenizer.split returns one piece per Unicode code point. BPE only merges bytes within a piece, so with every character in its own piece no merge ever happens. That is why the vocabulary stays at the 256 base byte values even though train is asked for 300, and why the tokenizer length prints as 256 rather than the 65 distinct characters in the text.

Because the base is bytes, any UTF-8 text round-trips, and a non-ASCII character simply becomes several byte tokens. Nothing is ever out of vocabulary.

The model code does not know any of this. Swap in a pre-tokenizer that splits on words, train to a larger vocabulary, and the same model trains on subword tokens, with shorter sequences. That one-line swap is the quickest way to see what BPE adds.

The key-value cache
Generation without a cache is O(T²): each new token re-runs the model over the whole context. A key-value cache stores the K and V projections from every past position, so each new token only needs to project its own K/V and attend against the stored ones. The cost drops to O(T).

The cache is implemented in kvcache.mojo:

KVCache stores (B, n_head, capacity, head_dim) tensors for K and V, plus a length counter and a capacity (= seq_length). It has four methods: reset, append, get, and evict_oldest.

get returns a view of the live prefix (B, n_head, length, head_dim), not a copy. The attention layer reads the cache without paying a per-step copy tax.

evict_oldest implements the sliding window: when the cache is full and a new token arrives, the oldest K/V is dropped and the rest shifted down. This is the cache-side equivalent of the T = min(total, n_ctx) crop in the uncached path.

The attention layer has two __call__ overloads:

__call__(x) — the training path. No cache, current behaviour.

__call__(x, ref cache) — the generation path. Computes K/V for the new positions only, appends to the cache, then attends the new query against the full cached K/V. Prefill (T_new > 1) uses the causal mask; decode (T_new = 1) has no mask, because the single new query sees every cached position in its past.

The cache is tested in tests/test_kvcache.mojo (7 tests) and the attention integration is tested in tests/test_attn_cache.mojo (3 tests, including a full token-by-token comparison against the uncached path). The generation path (generate_cached) is the next milestone. The oracle: cached generation must produce byte-identical output to uncached generation for the same seed, across four cases — no eviction, eviction, prompt length 1, and prompt longer than seq_length.

What it deliberately leaves out
GPU. Everything runs on CPU, to keep the code small and free of device plumbing.

Weight tying and checkpoint saving. Training starts from scratch every time, and the trained model is not saved.

Configuration files. Settings are key=value arguments; there is no config file.

Where it goes next
Planned, in order:

generate_cached. Wire the KV cache into the generation loop: prefill the prompt once, decode one token at a time. Validate against the uncached path (same seed, same output), then benchmark the speedup.

Ideas, not yet started:

Temperature and top-k sampling. Today generate samples from the model's raw distribution. Dividing the logits by a temperature below 1 makes the text more conservative, and keeping only the k most likely characters removes the unlikely tail, so samples contain fewer invented words. The loss does not change; only the sampling does.

A better training recipe. A learning-rate schedule (a short warmup, then decay) and gradient clipping. (Dropout is already a command-line argument via drop_ratio.)

Checkpoints. Save the trained weights, and add a separate generate command so a sample does not need a retrain.

A word-level pre-tokenizer. Compare characters against subword tokens on the same model.

A GPU path. A small, separate file, so the main code stays uncluttered.

Things to try
Predict first, then run.

Context. Run with seq_length=8, then 32, then 128. What happens to the loss and to the text? Context costs little compute at this width.

Overfit one batch. Train on a single fixed batch. The loss should fall close to zero, which proves the plumbing works.

Width and heads. Keep n_embd fixed and change n_head. Fewer, wider heads against more, narrower ones: which does the loss prefer?

Dropout. Change drop_ratio from 0.1 to 0.0 (or the other way) on the command line, with no recompile, and train for about 5 passes. Does the gap between train and validation loss change?

Position embeddings. Remove them. Can the model still use word order?

The mask. Remove the causal mask in MultiHeadAttention. Training loss will fall too well, because the model can read the answer; generation will not work. Why?

The pre-tokenizer. Swap CharPretokenizer for a word-splitting one and train to a larger vocabulary.

Sampling. Add a temperature to generate and compare samples from one trained model at 1.0, 0.8 and 0.5.

Credits
The Tiny Shakespeare text comes from Andrej Karpathy's char-rnn repository.

The structure follows Karpathy's nanoGPT and his video Let's build GPT.

Tenmo and mbpe are the libraries underneath; Tokens to Transformers in Mojo explains how they work.

License
MIT License. See LICENSE for details.
