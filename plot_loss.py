#!/usr/bin/env python3
"""plot_loss.py - turn a chargpt training log into loss_curve.svg.

Usage:
    python3 plot_loss.py                                   # newest run
    python3 plot_loss.py run_logs/chargpt_2026-10-26_10-37-54.log
    python3 plot_loss.py LOG -o my_curve.svg
    python3 plot_loss.py LOG --with-baseline               # include step 0 (loss ~6.7)

Default input is run_logs/chargpt_latest.log (next to this script). If that
symlink is missing (e.g. on Windows) the newest run_logs/chargpt_*.log is used.

Pure standard library; works anywhere Python 3.7+ runs. Works on a log that is
still being written, so you can re-run it during training.
"""

import argparse
import math
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

SCRIPT_DIR = Path(__file__).resolve().parent
STEP_RE = re.compile(
    r"\bstep\s+(\d+)\s*\|\s*train loss\s+(\S+)\s*\|\s*val loss\s+(\S+)"
)
HPARAMS_RE = re.compile(r"Hyper params:.*?\((.*)\)")
KV_RE = re.compile(r"(\w+)=([^,\s)]+)")
TOTAL_RE = re.compile(r"total wall clock\s*:\s*(.+)")

TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#d62728"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


# --------------------------------------------------------------- parsing --
def find_default_log():
    run_logs = SCRIPT_DIR / "run_logs"
    latest = run_logs / "chargpt_latest.log"
    if latest.exists():
        return latest
    candidates = sorted(
        p for p in run_logs.glob("chargpt_*.log") if "latest" not in p.name
    )
    return candidates[-1] if candidates else latest


def parse_log(path):
    steps, train, val = [], [], []
    hparams, wall = {}, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                try:
                    s, t, v = int(m.group(1)), float(m.group(2)), float(m.group(3))
                except ValueError:
                    continue
                if math.isfinite(t) and math.isfinite(v):
                    steps.append(s)
                    train.append(t)
                    val.append(v)
                continue
            m = HPARAMS_RE.search(line)
            if m:
                hparams = dict(KV_RE.findall(m.group(1)))
                continue
            m = TOTAL_RE.search(line)
            if m:
                wall = m.group(1).strip()
    return steps, train, val, hparams, wall


# --------------------------------------------------------------- scaling --
def nice_ticks(lo, hi, target=6):
    """Round tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if step >= raw:
            break
    start = math.floor(lo / step + 1e-9) * step
    ticks, v = [], start
    while v < hi + step * 0.999:
        ticks.append(round(v, 10))
        v += step
        if len(ticks) > 50:
            break
    return ticks, step


def fmt_tick(v, step):
    if step >= 1 and float(v).is_integer():
        return f"{int(v):,}".replace(",", "")
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return f"{v:.{decimals}f}"


# ------------------------------------------------------------------- svg --
def build_svg(steps, train, val, hparams, wall, run_name, zoomed):
    W, H = 960, 546
    L, R, T, B = 74, 86, 84, 96  # margins; plot area = W-L-R by H-T-B
    pw, ph = W - L - R, H - T - B
    bottom = T + ph

    planned = int(hparams["max_iters"]) if "max_iters" in hparams else 0
    xmax = max(planned, steps[-1])
    xticks, xstep = nice_ticks(0, xmax, 6)
    xticks = [t for t in xticks if t <= xmax * 1.0001]
    xmax_axis = xmax

    ymin, ymax = min(min(train), min(val)), max(max(train), max(val))
    pad = (ymax - ymin) * 0.06 or 0.1
    yticks, ystep = nice_ticks(ymin - pad, ymax + pad, 6)
    y_lo, y_hi = yticks[0], yticks[-1]

    def px(s):
        return L + s / xmax_axis * pw

    def py(v):
        return bottom - (v - y_lo) / (y_hi - y_lo) * ph

    o = []
    a = o.append
    a(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="{FONT}">'
    )
    a(f"<title>Training and validation loss - {escape(run_name)}</title>")
    a('<rect width="100%" height="100%" fill="white"/>')

    # Title block
    title = "Training and validation loss" + (" (zoomed)" if zoomed else "")
    a(
        f'<text x="{W / 2}" y="30" text-anchor="middle" font-size="18" '
        f'font-weight="bold" fill="#222">{escape(title)}</text>'
    )
    sub = []
    if "n_layer" in hparams:
        sub.append(
            f"{hparams.get('n_layer')} layers, {hparams.get('n_head', '?')} heads, "
            f"{hparams.get('n_embd', '?')}-dim"
        )
    if "batch_size" in hparams:
        sub.append(f"batch {hparams['batch_size']} x seq {hparams.get('seq_length', '?')}")
    if "learning_rate" in hparams:
        sub.append(f"lr {hparams['learning_rate']}")
    if "drop_ratio" in hparams:
        sub.append(f"dropout {hparams['drop_ratio']}")
    if sub:
        a(
            f'<text x="{W / 2}" y="50" text-anchor="middle" font-size="12" '
            f'fill="#666">{escape("  |  ".join(sub))}</text>'
        )
    a(
        f'<text x="{W / 2}" y="68" text-anchor="middle" font-size="11" '
        f'fill="#999">{escape(run_name)}</text>'
    )

    # Grid + tick labels
    for v in yticks:
        y = py(v)
        a(f'<line x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}" stroke="#ececec"/>')
        a(
            f'<text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="#555">{fmt_tick(v, ystep)}</text>'
        )
    for v in xticks:
        x = px(v)
        a(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{bottom}" stroke="#ececec"/>')
        a(
            f'<text x="{x:.1f}" y="{bottom + 18}" text-anchor="middle" font-size="11" '
            f'fill="#555">{fmt_tick(v, xstep)}</text>'
        )

    # Axes + labels
    a(f'<line x1="{L}" y1="{bottom}" x2="{L + pw}" y2="{bottom}" stroke="#333"/>')
    a(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{bottom}" stroke="#333"/>')
    a(
        f'<text x="{L + pw / 2}" y="{bottom + 42}" text-anchor="middle" '
        f'font-size="13" fill="#222">training step</text>'
    )
    ymid = T + ph / 2
    a(
        f'<text x="18" y="{ymid}" text-anchor="middle" font-size="13" fill="#222" '
        f'transform="rotate(-90 18 {ymid})">loss (nats)</text>'
    )

    # Curves
    def path(vals, color):
        pts = " L ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, vals))
        a(
            f'<path d="M {pts}" fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    path(train, TRAIN_COLOR)
    path(val, VAL_COLOR)

    # Points with hover tooltips (skipped for very dense logs)
    if len(steps) <= 80:
        for s, t, v in zip(steps, train, val):
            a(
                f'<circle cx="{px(s):.1f}" cy="{py(t):.1f}" r="3" fill="white" '
                f'stroke="{TRAIN_COLOR}" stroke-width="1.5">'
                f"<title>step {s}: train {t:.4f}</title></circle>"
            )
            a(
                f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="3" fill="white" '
                f'stroke="{VAL_COLOR}" stroke-width="1.5">'
                f"<title>step {s}: val {v:.4f}</title></circle>"
            )

    # Best validation point: ring marker
    bi = min(range(len(val)), key=lambda i: val[i])
    a(
        f'<circle cx="{px(steps[bi]):.1f}" cy="{py(val[bi]):.1f}" r="7" fill="none" '
        f'stroke="{VAL_COLOR}" stroke-width="1.5" stroke-dasharray="3 2">'
        f"<title>best val {val[bi]:.4f} at step {steps[bi]}</title></circle>"
    )

    # End-of-line labels (nudged apart if they would collide)
    yt, yv = py(train[-1]), py(val[-1])
    if abs(yt - yv) < 15:
        mid = (yt + yv) / 2
        yt, yv = (mid - 8, mid + 8) if yt <= yv else (mid + 8, mid - 8)
    lx = px(steps[-1]) + 10
    a(
        f'<text x="{lx:.1f}" y="{yt + 4:.1f}" font-size="12" font-weight="bold" '
        f'fill="{TRAIN_COLOR}">{train[-1]:.3f}</text>'
    )
    a(
        f'<text x="{lx:.1f}" y="{yv + 4:.1f}" font-size="12" font-weight="bold" '
        f'fill="{VAL_COLOR}">{val[-1]:.3f}</text>'
    )

    # Legend (top right, inside the plot where curves have already descended)
    lgx, lgy = L + pw - 150, T + 12
    a(
        f'<rect x="{lgx}" y="{lgy}" width="140" height="52" rx="4" fill="white" '
        f'fill-opacity="0.92" stroke="#ccc"/>'
    )
    a(f'<line x1="{lgx + 10}" y1="{lgy + 20}" x2="{lgx + 38}" y2="{lgy + 20}" '
      f'stroke="{TRAIN_COLOR}" stroke-width="2.5"/>')
    a(f'<text x="{lgx + 46}" y="{lgy + 24}" font-size="12" fill="#333">train loss</text>')
    a(f'<line x1="{lgx + 10}" y1="{lgy + 40}" x2="{lgx + 38}" y2="{lgy + 40}" '
      f'stroke="{VAL_COLOR}" stroke-width="2.5"/>')
    a(f'<text x="{lgx + 46}" y="{lgy + 44}" font-size="12" fill="#333">val loss</text>')

    # Footer: summary stats
    done = steps[-1]
    running = bool(planned and done < planned and not wall)
    parts = [
        f"{'latest' if running else 'final'} (step {done}): train {train[-1]:.3f}, val {val[-1]:.3f}",
        f"best val {val[bi]:.3f} @ step {steps[bi]}",
        f"gap {val[-1] - train[-1]:+.3f}",
    ]
    if running:
        parts.append(f"in progress: {done}/{planned} steps")
    if wall:
        parts.append(f"wall clock {wall}")
    a(
        f'<text x="{W / 2}" y="{H - 18}" text-anchor="middle" font-size="12" '
        f'fill="#444">{escape("   |   ".join(parts))}</text>'
    )
    a(
        f'<text x="{W / 2}" y="{H - 4}" text-anchor="middle" font-size="10" '
        f'fill="#aaa">dashed ring = best validation loss</text>'
    )

    a("</svg>")
    return "\n".join(o) + "\n"


# ------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser(description="Plot train/val loss from a chargpt log.")
    ap.add_argument("log", nargs="?", help="log file (default: run_logs/chargpt_latest.log)")
    ap.add_argument("-o", "--output", default="loss_curve.svg", help="output SVG (default: loss_curve.svg)")
    ap.add_argument(
        "--with-baseline",
        action="store_true",
        help="include the step-0 baseline (loss ~6.7) instead of zooming in",
    )
    args = ap.parse_args()

    log = Path(args.log) if args.log else find_default_log()
    if not log.is_file():
        sys.exit(f"error: log file not found: {log}")

    steps, train, val, hparams, wall = parse_log(log)
    if not args.with_baseline:
        keep = [i for i, s in enumerate(steps) if s > 0]
        steps = [steps[i] for i in keep]
        train = [train[i] for i in keep]
        val = [val[i] for i in keep]

    if not steps:
        sys.exit(
            f"error: no 'step N | train loss ... | val loss ...' lines found in {log}\n"
            "       (the first eval happens at eval_interval steps - try again later)"
        )

    svg = build_svg(
        steps, train, val, hparams, wall,
        run_name=log.resolve().name,
        zoomed=not args.with_baseline,
    )
    Path(args.output).write_text(svg, encoding="utf-8")
    print(f"Read {len(steps)} eval points from {log}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

