from std.pathlib import Path
from std.os.process import Process
from std.sys import argv

#from std.math import sqrt

comptime download_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
comptime input_file = "/tmp/input.txt"

#comptime INT = DType.int
#comptime INT64 = DType.int64
#comptime F32 = DType.float32

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
        var first = True
        var arguments = Dict[String, String]()
        for arg in args:
            if first:
                first = False
                continue
            var key_value = arg.split("=")
            if key_value[0] in arguments:
                continue
            arguments[String(key_value[0])] = String(key_value[1])
        print(arguments)

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
        self.seed_range = (
            UInt64(
                Self.to_dtype(arguments["seed_range"], Int(seed_range))
            ) if "seed_range"
            in arguments else seed_range
        )
        self.gen_interval = (
            Self.to_dtype(arguments["gen_interval"], gen_interval) if "gen_interval"
            in arguments else gen_interval
        )
        self.learning_rate = (
            Self.to_dtype(arguments["learning_rate"], learning_rate) if "learning_rate"
            in arguments else learning_rate
        )
        self.drop_ratio = (
            Self.to_dtype(arguments["drop_ratio"], drop_ratio) if "drop_ratio"
            in arguments else drop_ratio
        )

    @staticmethod
    def to_dtype[dtype: DType, //](s: String, value: Scalar[dtype]) -> Scalar[dtype]:
        try:
            comptime if dtype == DType.int:
                return Scalar[dtype](Int(s))
            else:
                return Scalar[dtype](Float64(s))
        except:
            return value


struct TinyShakespeare:
    @staticmethod
    def text() raises -> String:
        var path = Path(input_file)
        if not path.exists() or not path.is_file():
            print("Downloading tiny shakespeare (~1MB)...")
            var args = List[String]()
            args.append("-P")
            args.append("/tmp")
            args.append(download_url)
            var process = Process.run("wget", args^)
            _ = process.wait()
            print("Tiny shakespeare downloaded")

        var content = path.read_text()
        print("Total characters: ", content.byte_length())
        print("Total code points: ", content.count_codepoints())
        print("Total graphemes: ", content.count_graphemes())
        print()
        return content^
