# =============================================================================
# shared.mojo - what bigram.mojo and chargpt.mojo have in common.
#
#   CmdLineArgs      every run-time setting, read from key=value arguments
#   TinyShakespeare  fetches the training text once and returns it as a String
#
# Why this file exists: a `comptime` constant is baked in at compile time, so
# changing one means recompiling the whole program. Settings that are read
# from the command line at run time can be changed without a recompile:
#
#   mojo chargpt.mojo batch_size=16 seq_length=32 n_embd=96
# =============================================================================

from std.pathlib import Path
from std.os.process import Process
from std.sys import argv

# Where the text comes from, and where the downloaded copy is kept. Once the
# file exists in /tmp, later runs reuse it and do not download again.
comptime download_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
comptime input_file = "/tmp/input.txt"


# -----------------------------------------------------------------------------
# CmdLineArgs: all run-time settings in one place
# -----------------------------------------------------------------------------
# Each field below has a default, given in `__init__`. Anything not named on
# the command line keeps its default.
#
#   field            meaning
#   ---------------  -------------------------------------------------------
#   batch_size       windows of text per training step
#   seq_length       characters per window (the context length)
#   eval_iters       batches averaged for one loss reading
#   eval_interval    steps between loss readings
#   max_iters        total training steps
#   max_new_tokens   characters per generated sample
#   n_embd           width of every vector inside the model
#   n_head           attention heads (n_embd must divide evenly by it)
#   n_layer          number of transformer blocks
#   seed_range       upper bound for the random seeds used when sampling
#   gen_interval     steps between samples
#   learning_rate    AdamW step size
#   drop_ratio       dropout probability (0.0 = off)
#
# `ImplicitlyCopyable` lets a CmdLineArgs be passed around by value without
# an explicit copy; `Writable` lets `print(h_params)` show every field.
@fieldwise_init
struct CmdLineArgs(ImplicitlyCopyable, Writable):
    var batch_size: Int
    var seq_length: Int
    var eval_iters: Int
    var eval_interval: Int
    var max_iters: Int
    var max_new_tokens: Int
    var n_embd: Int
    var n_head: Int
    var n_layer: Int
    var seed_range: UInt64
    var gen_interval: Int
    var learning_rate: Float32
    var drop_ratio: Float32

    def __init__(
        out self,
        args: Span[StringSpan[ImmStaticOrigin], ImmStaticOrigin],
        # The defaults. Each one is used when its key is absent from `args`
        # (or when its value cannot be parsed, see `to_dtype`).
        batch_size: Int = 32,
        seq_length: Int = 32,
        eval_iters: Int = 50,
        eval_interval: Int = 300,
        max_iters: Int = 10000,
        max_new_tokens: Int = 500,
        n_embd: Int = 192,
        n_head: Int = 6,
        n_layer: Int = 6,
        seed_range: UInt64 = 1000000,
        gen_interval: Int = 500,
        learning_rate: Float32 = 1e-3,
        drop_ratio: Float32 = 0.1,
    ) raises:
        # Step 1: turn ["prog", "a=1", "b=2"] into the dictionary
        # {"a": "1", "b": "2"}.
        var first = True
        var arguments = Dict[String, String]()
        for arg in args:
            # The first entry is the program name, not a setting.
            if first:
                first = False
                continue
            # "key=value" -> ["key", "value"]
            var key_value = arg.split("=")
            # If a key is repeated, the first occurrence wins.
            if key_value[0] in arguments:
                continue
            arguments[String(key_value[0])] = String(key_value[1])
        print(arguments)

        # Step 2: for each field, use the command-line value if one was
        # given, otherwise the default. The pattern is the same every time:
        #
        #   self.x = parse(arguments["x"], default) if "x" in arguments
        #            else default
        #
        # `to_dtype` falls back to the default when the text is not a valid
        # number, so a typo in a value is silently ignored. A misspelt KEY is
        # ignored too, because nothing ever looks it up. Check the printed
        # "Hyper params" line to see what was actually used.
        self.batch_size = (
            Self.to_dtype(arguments["batch_size"], batch_size) if "batch_size"
            in arguments else batch_size
        )
        self.seq_length = (
            Self.to_dtype(arguments["seq_length"], seq_length) if "seq_length"
            in arguments else seq_length
        )
        self.eval_iters = (
            Self.to_dtype(arguments["eval_iters"], eval_iters) if "eval_iters"
            in arguments else eval_iters
        )
        self.eval_interval = (
            Self.to_dtype(
                arguments["eval_interval"], eval_interval
            ) if "eval_interval"
            in arguments else eval_interval
        )
        self.max_iters = (
            Self.to_dtype(arguments["max_iters"], max_iters) if "max_iters"
            in arguments else max_iters
        )

        self.max_new_tokens = (
            Self.to_dtype(
                arguments["max_new_tokens"], max_new_tokens
            ) if "max_new_tokens"
            in arguments else max_new_tokens
        )
        self.n_embd = (
            Self.to_dtype(arguments["n_embd"], n_embd) if "n_embd"
            in arguments else n_embd
        )

        self.n_head = (
            Self.to_dtype(arguments["n_head"], n_head) if "n_head"
            in arguments else n_head
        )
        self.n_layer = (
            Self.to_dtype(arguments["n_layer"], n_layer) if "n_layer"
            in arguments else n_layer
        )
        # seed_range is unsigned 64-bit: parse it as an Int, convert after.
        self.seed_range = (
            UInt64(
                Self.to_dtype(arguments["seed_range"], Int(seed_range))
            ) if "seed_range"
            in arguments else seed_range
        )
        self.gen_interval = (
            Self.to_dtype(
                arguments["gen_interval"], gen_interval
            ) if "gen_interval"
            in arguments else gen_interval
        )
        self.learning_rate = (
            Self.to_dtype(
                arguments["learning_rate"], learning_rate
            ) if "learning_rate"
            in arguments else learning_rate
        )
        self.drop_ratio = (
            Self.to_dtype(arguments["drop_ratio"], drop_ratio) if "drop_ratio"
            in arguments else drop_ratio
        )

    # Parse `s` as the same type as `value`, which is also the fallback.
    #   - Int-typed settings (dtype is DType.int) are parsed as integers.
    #   - Everything else (the Float32 settings) is parsed as a float.
    # If parsing fails, `value` (the default) comes back unchanged and no
    # error is shown. `comptime if` chooses the branch at compile time, so
    # each call compiles to just one of the two parsers.
    @staticmethod
    def to_dtype[
        dtype: DType, //
    ](s: String, value: Scalar[dtype]) -> Scalar[dtype]:
        try:
            comptime if dtype == DType.int:
                return Scalar[dtype](Int(s))
            else:
                return Scalar[dtype](Float64(s))
        except:
            return value


# -----------------------------------------------------------------------------
# TinyShakespeare: the training text
# -----------------------------------------------------------------------------
# About 1 MB of Shakespeare plays (the file from Karpathy's char-rnn repo).
# It is downloaded with `wget` the first time and cached in /tmp.
struct TinyShakespeare:
    @staticmethod
    def text() raises -> String:
        var path = Path(input_file)
        # Download only if the cached copy is missing.
        if not path.exists() or not path.is_file():
            print("Downloading tiny shakespeare (~1MB)...")
            # Equivalent to: wget -P /tmp <download_url>
            # (-P sets the directory the file is saved into.)
            var args = List[String]()
            args.append("-P")
            args.append("/tmp")
            args.append(download_url)
            var process = Process.run("wget", args^)
            _ = process.wait()
            print("Tiny shakespeare downloaded")

        var content = path.read_text()
        # Three ways to count the same text. For plain ASCII all
        # three are equal; they differ for accented letters or emoji, which
        # take several bytes, or several code points, each.
        print("Total bytes: ", content.byte_length())
        print("Total code points: ", content.count_codepoints())
        print("Total graphemes: ", content.count_graphemes())
        print()
        return content^
