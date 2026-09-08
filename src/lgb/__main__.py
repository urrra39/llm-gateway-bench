"""Command-line entry point.

Stages, in order: workload -> tune -> run (baseline/cache/router) -> judge ->
metrics. Every stage is resumable and safe to rerun.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lgb.config import Config

DEFAULT_CONFIG = Path("config/bench.yaml")


def _cfg(args: argparse.Namespace) -> Config:
    return Config.load(args.config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lgb")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("workload", help="build the two duplicate-fraction workloads")
    sub.add_parser("tune", help="choose the cache threshold on the tuning half")

    run = sub.add_parser("run", help="execute one config on one workload fraction")
    run.add_argument(
        "--config",
        action="store",
        dest="run_config",
        required=True,
        choices=("baseline", "cache", "router_cascade", "router_heuristic"),
    )
    run.add_argument("--frac", required=True, choices=("low", "high"))
    run.add_argument("--limit", type=int, default=None, help="stop after N rows (pilot)")

    judge = sub.add_parser("judge", help="judge one config's responses against baseline")
    judge.add_argument(
        "--config",
        action="store",
        dest="run_config",
        required=True,
        choices=("baseline", "cache", "router_cascade", "router_heuristic"),
    )
    judge.add_argument("--frac", required=True, choices=("low", "high"))

    metrics = sub.add_parser("metrics", help="assemble the primary results file")
    metrics.add_argument("--name", required=True)
    metrics.add_argument("--note", default="")

    sub.add_parser("tune-report", help="print the tuning record")
    sub.add_parser("results", help="print the current primary results file")

    exp = sub.add_parser("export-human", help="write the human-validation CSV")
    exp.add_argument("--out", type=Path, default=Path("data/human_validation.csv"))
    exp.add_argument("--limit", type=int, default=60)

    ha = sub.add_parser("human-agreement", help="judge-vs-human agreement from a filled CSV")
    ha.add_argument("--csv", type=Path, required=True)

    sub.add_parser("serve", help="run the FastAPI gateway")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if not args.command:
        parser.print_help()
        return 2

    from lgb import metrics as metrics_mod
    from lgb import run as run_mod

    if args.command == "workload":
        run_mod.build_all_workloads(_cfg(args))
        print("workloads built under data/runs/workloads/")
        return 0
    if args.command == "tune":
        from lgb.embed import Embedder

        cfg = _cfg(args)
        embedder = Embedder(cfg.embeddings.model, cfg.embeddings.models_dir)
        record = run_mod.tune_threshold(cfg, embedder)
        print(record)
        return 0
    if args.command == "run":
        from lgb.embed import Embedder

        cfg = _cfg(args)
        embedder = Embedder(cfg.embeddings.model, cfg.embeddings.models_dir)
        threshold = cfg.cache.sim_threshold
        tuning = run_mod.read_json(cfg.data.runs_dir / "tuning.json")
        if tuning and "threshold" in tuning:
            threshold = float(tuning["threshold"])
        frame = run_mod.execute_config(
            cfg,
            args.run_config,
            args.frac,
            threshold,
            embedder,
            run_mod.run_dir(cfg, args.run_config, args.frac),
            limit=args.limit,
        )
        print(f"wrote {len(frame)} outcomes to {run_mod.run_dir(cfg, args.run_config, args.frac)}")
        return 0
    if args.command == "judge":
        from lgb import judge as judge_mod

        cfg = _cfg(args)
        baseline = read_outcomes(cfg, "baseline", args.frac)
        outcomes = read_outcomes(cfg, args.run_config, args.frac)
        run_path = run_mod.run_dir(cfg, args.run_config, args.frac)
        judge_mod.judge_run(cfg, args.run_config, args.frac, baseline, outcomes, run_path)
        print(f"judged {args.run_config} {args.frac}")
        return 0
    if args.command == "metrics":
        payload = metrics_mod.assemble(_cfg(args), args.name, args.note)
        print("all_gates_passed:", payload["all_gates_passed"])
        for g in payload["gates"]:
            print(("PASS" if g["passed"] else "FAIL"), g["name"], g["observed"])
        return 0
    if args.command == "tune-report":
        from lgb.store import read_json

        cfg = _cfg(args)
        record = read_json(cfg.data.runs_dir / "tuning.json")
        print(record)
        return 0
    if args.command == "results":
        cfg = _cfg(args)
        from lgb.store import read_json

        print(read_json(cfg.data.primary_results))
        return 0
    if args.command == "export-human":
        from lgb import humanval

        cfg = _cfg(args)
        path = humanval.export_human_csv(cfg, args.out, args.limit)
        print(f"wrote {path}")
        return 0
    if args.command == "human-agreement":
        from lgb import humanval

        print(humanval.human_agreement(args.csv))
        return 0
    if args.command == "serve":
        import uvicorn

        from lgb.api import build_app

        cfg = _cfg(args)
        uvicorn.run(build_app(cfg), host="127.0.0.1", port=8000)
        return 0
    return 2  # pragma: no cover


def read_outcomes(cfg: Config, config: str, frac: str):
    from lgb.run import run_dir
    from lgb.store import read_parquet

    path = run_dir(cfg, config, frac) / "outcomes.parquet"
    frame = read_parquet(path)
    if frame is None:
        raise FileNotFoundError(f"no outcomes at {path}; run `lgb run` first")
    return frame


if __name__ == "__main__":
    raise SystemExit(main())
