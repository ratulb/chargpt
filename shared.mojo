"""What bigram.mojo and chargpt.mojo share?

   CmdLineArgs      every run-time setting, read from key=value arguments
   TinyShakespeare  fetches the training text once and returns it as a String

   A `comptime` constant is baked in at compile time, so changing one means
   recompiling the whole program. Settings that are read from the command line
   at run time can be changed without a recompile:

   mojo chargpt.mojo batch_size=16 seq_length=32 n_embd=96."""

from std.pathlib import Path
from std.os.process import Process
from std.sys import argv

# Where the text comes from, and where the downloaded copy is kept. Once the
# file exists in /tmp, later runs reuse it and do not download again.
comptime download_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
comptime input_file = "/tmp/input.txt"


@fieldwise_init
struct CmdLineArgs(ImplicitlyCopyable, Writable):
    """
    CmdLineArgs: all run-time settings in one place
    Each field below has a default, given in `__init__`. Anything not named on
    the command line keeps its default.

    field            meaning
    ---------------  -------
    batch_size       windows of text per training step
    seq_length       characters per window (the context length)
    eval_iters       batches averaged for one loss reading
    eval_interval    steps between loss readings
    max_iters        total training steps
    max_new_tokens   characters per generated sample
    n_embd           width of every vector inside the model
    n_head           attention heads (n_embd must divide evenly by it)
    n_layer          number of transformer blocks
    learning_rate    AdamW step size
    drop_ratio       dropout probability (0.0 = off)

    `ImplicitlyCopyable` lets a CmdLineArgs be passed around by value without
    an explicit copy; `Writable` lets `print(h_params)` show every field.

    """

    var batch_size: Int
    var seq_length: Int
    var eval_iters: Int
    var eval_interval: Int
    var max_iters: Int
    var max_new_tokens: Int
    var n_embd: Int
    var n_head: Int
    var n_layer: Int
    var learning_rate: Float32
    var drop_ratio: Float32

    def __init__(
        out self,
        args: Span[StringSpan[ImmStaticOrigin], ImmStaticOrigin],
        # The defaults. Each one is used when its key is absent from `args`
        # (or when its value cannot be parsed, see `to_dtype`).
        batch_size: Int = 16,
        seq_length: Int = 32,
        eval_iters: Int = 50,
        eval_interval: Int = 300,
        max_iters: Int = 5000,
        max_new_tokens: Int = 500,
        n_embd: Int = 192,
        n_head: Int = 6,
        n_layer: Int = 6,
        learning_rate: Float32 = 1e-3,
        drop_ratio: Float32 = 0.1,
    ) raises:
        # Turn ["chargpt.mojo", "n_embd=192", "n_layer=6"] into the
        # dictionary {"n_embd": "192", "n_laywer": "6"}.
        var first = True
        var arguments = Dict[String, String]()
        for arg in args:
            # The first entry is the program name, not a setting.
            if first:
                first = False
                continue
            # "n_embd=192" -> ["n_embd", "192"]
            var key_value = arg.split("=")
            # If a key is repeated, the first occurrence wins.
            if key_value[0] in arguments:
                continue
            arguments[String(key_value[0])] = String(key_value[1])

        # For each field, use the command-line value if one was given,
        # otherwise the default. The pattern is the same every time:
        #
        # self.x = parse(arguments["x"], default) if "x" in arguments else default
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

    @staticmethod
    def to_dtype[
        dtype: DType, //
    ](s: String, value: Scalar[dtype]) -> Scalar[dtype]:
        """Parse `s` as the same type as `value`, which is also the fallback.
        - Int-typed settings (dtype is DType.int) are parsed as integers.
        - Everything else (the Float32 settings) is parsed as a float.
        If parsing fails, `value` (the default) comes back unchanged and no
        error is shown. `comptime if` chooses the branch at compile time, so
        each call compiles to just one of the two parsers."""

        try:
            comptime if dtype == DType.int:
                return Scalar[dtype](Int(s))
            else:
                # We return DType.float32 - but Mojo does not parse Float32!
                return Scalar[dtype](Float64(s))
        except:
            return value


struct TinyShakespeare:
    """
    TinyShakespeare: the training text
    About 1 MB of Shakespeare plays (the file from Karpathy's char-rnn repo).
    It is downloaded with `wget` (or `curl` if wget is missing) the first
    time and cached in /tmp.
    """

    @staticmethod
    def run_downloader(tool: String, var args: List[String]) -> Bool:
        """Run a downloader; True only if the cached file exists afterwards.
        A missing binary or a failed run is treated as 'not downloaded'."""
        try:
            var process = Process.run(tool, args^)
            _ = process.wait()
        except:
            return False
        var path = Path(input_file)
        return path.exists() and path.is_file()

    @staticmethod
    def text() raises -> String:
        var path = Path(input_file)
        if not path.exists() or not path.is_file():
            print("Downloading tiny shakespeare (~1MB)...")

            # Try with wget: wget -P /tmp <download_url>
            var wget_args = List[String]()
            wget_args.append("-P")
            wget_args.append("/tmp")
            wget_args.append(download_url)
            var ok = Self.run_downloader("wget", wget_args^)

            # Fallback: curl -fsSL -o <input_file> <download_url>
            #   -f  fail on HTTP errors instead of saving the error page
            #   -s  silent, -S show errors anyway, -L follow redirects
            if not ok:
                print("wget unavailable or failed, trying curl...")
                var curl_args = List[String]()
                curl_args.append("-fsSL")
                curl_args.append("-o")
                curl_args.append(input_file)
                curl_args.append(download_url)
                ok = Self.run_downloader("curl", curl_args^)

            if not ok:
                raise Error(
                    "Could not download tiny shakespeare. Install wget or"
                    " curl, or place the file at "
                    + input_file
                )
            print("Tiny shakespeare downloaded")

        var content = path.read_text()
        print("Total bytes: ", content.byte_length())
        print("Total code points: ", content.count_codepoints())
        print("Total graphemes: ", content.count_graphemes())
        print()
        return content^
