"""Generate SVG figures from results.json with stdlib only."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results.json"
FIGS = REPO / "docs" / "figures"


def _bar_svg(title: str, series: dict[str, dict[str, float]], ylabel: str) -> str:
    configs = ["baseline", "cache", "router_cascade", "router_heuristic"]
    fracs = ["low", "high"]
    vals = [series[f][c] for f in fracs for c in configs]
    max_v = max(vals) if vals else 0.0
    w, h, pad = 560, 320, 50
    bar_w = 40
    gap = 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    out.append(f"<text x='{w//2}' y='24' text-anchor='middle' font-size='14'>{title}</text>")
    out.append(
        f"<text x='14' y='{h//2}' font-size='11' "
        f"transform='rotate(-90 14 {h//2})'>{ylabel}</text>"
    )
    colors = {"low": "#4a7c9b", "high": "#c26a4a"}
    for fi, frac in enumerate(fracs):
        for ci, cfg in enumerate(configs):
            v = series[frac][cfg]
            x = pad + ci * (bar_w * 2 + gap) + fi * bar_w
            bh = (h - pad - 40) * (v / max_v) if max_v else 0
            y = h - 40 - bh
            out.append(
                f"<rect x='{x:.0f}' y='{y:.0f}' width='{bar_w}' height='{bh:.0f}' "
                f"fill='{colors[frac]}'><title>{frac} {cfg}: {v:.4f}</title></rect>"
            )
            if fi == 0:
                out.append(
                    f"<text x='{x + bar_w:.0f}' y='{h - 22}' text-anchor='middle' "
                    f"font-size='9'>{cfg.replace('router_', 'r_')}</text>"
                )
    out.append("<text x='500' y='60' font-size='11' fill='#4a7c9b'>low</text>")
    out.append("<text x='500' y='76' font-size='11' fill='#c26a4a'>high</text>")
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    cost = {
        f: {c: float(metrics[f"{f}_{c}"]["cost_usd"]) for c in
            ["baseline", "cache", "router_cascade", "router_heuristic"]}
        for f in ["low", "high"]
    }
    quality = {
        f: {c: (metrics[f"{f}_{c}"].get("quality_equiv") or 0.0) for c in
            ["baseline", "cache", "router_cascade", "router_heuristic"]}
        for f in ["low", "high"]
    }
    # Baseline has no quality; show 1.0 reference for scale honesty.
    for f in ["low", "high"]:
        quality[f]["baseline"] = 1.0
    (FIGS / "cost_by_config.svg").write_text(
        _bar_svg("Assumed cost by config (USD)", cost, "USD"), encoding="utf-8"
    )
    (FIGS / "quality_by_config.svg").write_text(
        _bar_svg("Quality equiv by config", quality, "equiv"), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
