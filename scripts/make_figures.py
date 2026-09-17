"""Generate SVG figures from results.json and run parquet with stdlib only.

Display labels name the recipe, not a price tier that does not exist: both
tiers are operating points of deepseek-v4-flash. Machine config IDs from
results.json are shown in parentheses so figures map back to the tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results.json"
RUNS = REPO / "data" / "runs"
FIGS = REPO / "docs" / "figures"

CONFIGS = ["baseline", "cache", "router_cascade", "router_heuristic"]
FRACS = ["low", "high"]

#: Honest display names: what each configuration actually runs. The machine
#: config ID follows in parentheses wherever space allows.
DISPLAY = {
    "baseline": "long only",
    "cache": "cache + long",
    "router_cascade": "cache + cascade",
    "router_heuristic": "cache + heuristic",
}

COLORS = {
    "baseline": "#555555",
    "cache": "#4a7c9b",
    "router_cascade": "#c26a4a",
    "router_heuristic": "#6a9b4a",
}


def _bar_svg(title: str, series: dict[str, dict[str, float]], ylabel: str) -> str:
    vals = [series[f][c] for f in FRACS for c in CONFIGS]
    max_v = max(vals) if vals else 0.0
    w, h, pad = 640, 340, 50
    bar_w = 44
    gap = 28
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    out.append(f"<text x='{w//2}' y='24' text-anchor='middle' font-size='14'>{title}</text>")
    out.append(
        f"<text x='14' y='{h//2}' font-size='11' "
        f"transform='rotate(-90 14 {h//2})'>{ylabel}</text>"
    )
    colors = {"low": "#4a7c9b", "high": "#c26a4a"}
    for fi, frac in enumerate(FRACS):
        for ci, cfg in enumerate(CONFIGS):
            v = series[frac][cfg]
            x = pad + ci * (bar_w * 2 + gap) + fi * bar_w
            bh = (h - pad - 60) * (v / max_v) if max_v else 0
            y = h - 60 - bh
            out.append(
                f"<rect x='{x:.0f}' y='{y:.0f}' width='{bar_w}' height='{bh:.0f}' "
                f"fill='{colors[frac]}'><title>{frac} {cfg}: {v:.4f}</title></rect>"
            )
            if fi == 0:
                out.append(
                    f"<text x='{x + bar_w:.0f}' y='{h - 42}' text-anchor='middle' "
                    f"font-size='10'>{DISPLAY[cfg]}</text>"
                )
                out.append(
                    f"<text x='{x + bar_w:.0f}' y='{h - 28}' text-anchor='middle' "
                    f"font-size='8' fill='#666'>({cfg})</text>"
                )
    out.append("<text x='560' y='60' font-size='11' fill='#4a7c9b'>low dup</text>")
    out.append("<text x='560' y='76' font-size='11' fill='#c26a4a'>high dup</text>")
    out.append("</svg>")
    return "\n".join(out)


def _line_svg(
    title: str,
    xs: list[float],
    lines: dict[str, list[float]],
    ylabel: str,
    mark_x: float | None = None,
) -> str:
    w, h = 640, 360
    ml, mr, mt, mb = 56, 16, 36, 44
    colors = {"recall": "#4a7c9b", "false_hit_rate": "#c0392b", "hit_rate": "#6a9b4a"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    out.append(f"<text x='{w//2}' y='22' text-anchor='middle' font-size='14'>{title}</text>")
    out.append(
        f"<text x='14' y='{(mt + h - mb) / 2:.0f}' font-size='11' "
        f"transform='rotate(-90 14 {(mt + h - mb) / 2:.0f})'>{ylabel}</text>"
    )
    lo, hi = min(xs), max(xs)
    all_y = [v for vals in lines.values() for v in vals]
    ymax = max(all_y) if all_y else 1.0

    def px(x: float) -> float:
        return ml + (x - lo) / (hi - lo) * (w - ml - mr)

    def py(v: float) -> float:
        return h - mb - v / ymax * (h - mt - mb)

    for gy in (0.0, 0.25, 0.5, 0.75, 1.0):
        out.append(
            f"<line x1='{ml}' y1='{py(gy):.1f}' x2='{w - mr}' y2='{py(gy):.1f}' "
            "stroke='#ddd'/>"
        )
        out.append(f"<text x='{ml - 6}' y='{py(gy) + 4:.1f}' text-anchor='end' "
                   f"font-size='9'>{gy:.2f}</text>")
    if mark_x is not None:
        out.append(
            f"<line x1='{px(mark_x):.1f}' y1='{mt}' x2='{px(mark_x):.1f}' "
            f"y2='{h - mb}' stroke='#333' stroke-dasharray='4,3'/>"
        )
        out.append(
            f"<text x='{px(mark_x):.1f}' y='{mt - 6}' text-anchor='middle' "
            f"font-size='10'>chosen {mark_x}</text>"
        )
    for li, (name, vals) in enumerate(lines.items()):
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(xs, vals, strict=True))
        out.append(
            f"<polyline points='{pts}' fill='none' stroke='{colors[name]}' "
            f"stroke-width='2'><title>{name}</title></polyline>"
        )
        out.append(
            f"<text x='{w - mr}' y='{mt + 14 * li}' text-anchor='end' font-size='11' "
            f"fill='{colors[name]}'>{name}</text>"
        )
    out.append(
        f"<text x='{(ml + w - mr) / 2:.0f}' y='{h - 22}' text-anchor='middle' "
        "font-size='10'>threshold 0.79 chosen for F1; no threshold reaches 2% false hits"
        "</text>"
    )
    out.append(
        f"<text x='{(ml + w - mr) / 2:.0f}' y='{h - 8}' text-anchor='middle' "
        "font-size='10'>similarity threshold</text>"
    )
    return "\n".join(out) + "\n</svg>\n"


def _hist_svg(title: str, frac: str, series: dict[str, list[float]]) -> str:
    """Overlaid per-config latency histograms on a log time axis."""
    w, h = 680, 360
    ml, mr, mt, mb = 56, 16, 40, 60
    lo_l, hi_l = math.log10(0.5), math.log10(100000.0)
    nbins = 28
    counts: dict[str, list[int]] = {}
    for cfg, vals in series.items():
        bins = [0] * nbins
        for v in vals:
            b = int((math.log10(max(v, 0.5)) - lo_l) / (hi_l - lo_l) * nbins)
            bins[min(max(b, 0), nbins - 1)] += 1
        counts[cfg] = bins
    cmax = max(max(b) for b in counts.values())
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    out.append(f"<text x='{w//2}' y='22' text-anchor='middle' font-size='14'>{title}</text>")

    def px(b: int) -> float:
        return ml + b / nbins * (w - ml - mr)

    def py(c: int) -> float:
        return h - mb - (c / cmax) * (h - mt - mb) if cmax else h - mb

    bw = (w - ml - mr) / nbins
    for b in range(nbins):
        for cfg in CONFIGS:
            c = counts[cfg][b]
            if not c:
                continue
            out.append(
                f"<rect x='{px(b):.1f}' y='{py(c):.1f}' width='{bw - 1:.1f}' "
                f"height='{h - mb - py(c):.1f}' fill='{COLORS[cfg]}' "
                f"fill-opacity='0.45'><title>{cfg} bin {b}: {c}</title></rect>"
            )
    for ms, label in ((1, "1ms"), (100, "100ms"), (10000, "10s")):
        x = ml + (math.log10(ms) - lo_l) / (hi_l - lo_l) * (w - ml - mr)
        out.append(
            f"<line x1='{x:.1f}' y1='{mt}' x2='{x:.1f}' y2='{h - mb}' stroke='#ddd'/>"
        )
        out.append(
            f"<text x='{x:.1f}' y='{h - mb + 14}' text-anchor='middle' "
            f"font-size='9'>{label}</text>"
        )
    for li, cfg in enumerate(CONFIGS):
        out.append(
            f"<text x='{w - mr}' y='{mt + 14 * li}' text-anchor='end' font-size='11' "
            f"fill='{COLORS[cfg]}'>{DISPLAY[cfg]} ({cfg})</text>"
        )
    out.append(
        f"<text x='{(ml + w - mr) / 2:.0f}' y='{h - 22}' text-anchor='middle' "
        f"font-size='10'>warm per-request latency, {frac} duplicate fraction "
        "(log time axis; cache hits cluster left, model-bound tail right)</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    import pandas as pd

    FIGS.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    cost = {
        f: {c: float(metrics[f"{f}_{c}"]["cost_usd"]) for c in CONFIGS} for f in FRACS
    }
    quality = {
        f: {c: (metrics[f"{f}_{c}"].get("quality_equiv") or 0.0) for c in CONFIGS}
        for f in FRACS
    }
    # Baseline has no quality; show 1.0 reference for scale honesty.
    for f in FRACS:
        quality[f]["baseline"] = 1.0
    (FIGS / "cost_by_config.svg").write_text(
        _bar_svg("Assumed cost by config (USD)", cost, "USD"), encoding="utf-8"
    )
    (FIGS / "quality_by_config.svg").write_text(
        _bar_svg("Quality equiv by config", quality, "equiv"), encoding="utf-8"
    )
    sweep = pd.read_parquet(RUNS / "threshold_sweep.parquet").sort_values("threshold")
    elig = float((sweep.tp + sweep.fp + sweep.fn + sweep.tn).iloc[0])
    xs = [float(x) for x in sweep.threshold.tolist()]
    (FIGS / "threshold_tradeoff.svg").write_text(
        _line_svg(
            "Threshold trade-off on the tuning half",
            xs,
            {
                "recall": [float(v) for v in sweep.recall.tolist()],
                "false_hit_rate": [float(v) for v in sweep.false_hit_rate.tolist()],
                "hit_rate": [
                    float((t + f) / elig) for t, f in zip(sweep.tp, sweep.fp, strict=True)
                ],
            },
            "rate",
            mark_x=0.79,
        ),
        encoding="utf-8",
    )
    for frac in FRACS:
        series: dict[str, list[float]] = {}
        for cfg in CONFIGS:
            frame = pd.read_parquet(RUNS / f"{frac}_{cfg}" / "outcomes.parquet")
            ok = frame[frame["error"].isna()]
            warm = ok.iloc[1:]
            series[cfg] = [float(v) for v in warm["latency_ms"].tolist()]
        (FIGS / f"latency_{frac}.svg").write_text(
            _hist_svg(f"Latency distribution ({frac} duplicates)", frac, series),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
