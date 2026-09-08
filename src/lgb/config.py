"""Pydantic config loaded from a single YAML. Every tunable lives here."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GatewaySpec(Frozen):
    base_url: str = "http://127.0.0.1:8787/v1"
    api_key_env: str | None = "GSK_API_KEY"
    timeout_s: float = 120.0
    concurrency: int = 4
    max_retries: int = 3


class GenerationSpec(Frozen):
    temperature: float = 0.0
    max_tokens: int = 4096
    #: Both gateway models are reasoning models on the dev gateway: they emit a
    #: chain of thought into `reasoning_content` before the answer, and a small
    #: max_tokens can be consumed entirely by reasoning, leaving `content`
    #: empty. A low reasoning effort keeps answers fast and non-empty while
    #: still letting the model think briefly. Measured 2026-09-08.
    reasoning_effort: str | None = "low"
    system_prompt: str = "Answer the request concisely, in at most three sentences. Do not refuse."


class ModelSpec(Frozen):
    name: str


class ModelsSpec(Frozen):
    cheap: str
    expensive: str
    judge_primary: str
    judge_secondary: str


class PriceSpec(Frozen):
    input_per_mtok: float
    output_per_mtok: float


class PricesSpec(Frozen):
    currency: str = "USD"
    as_of: str = ""
    basis: str = ""
    per_million_tokens: dict[str, PriceSpec]


class EmbeddingsSpec(Frozen):
    provider: Literal["local_sentence_transformers", "api"] = "local_sentence_transformers"
    model: str = "all-MiniLM-L6-v2"
    models_dir: Path = Path("data/models")
    batch_size: int = 64


class CacheSpec(Frozen):
    exact_match: bool = True
    sim_threshold: float = 0.86
    #: Fraction of the workload reserved for threshold tuning; the reported run
    #: is the held-out half. Must be in (0, 1).
    tune_fraction: float = 0.5
    tune_seed: int = 20260908


class HeuristicSpec(Frozen):
    max_chars: int = 240
    #: "explain" is intentionally absent: the workload wrapper makes it constant.
    reasoning_cues: tuple[str, ...] = (
        "why",
        "compare",
        "steps",
        "reason",
        "because",
        "how",
        "solve",
        "calculate",
        "difference",
        "prove",
        "justify",
    )


class RouterSpec(Frozen):
    heuristic: HeuristicSpec
    #: Cascade escalates to the expensive model when the cheap answer contains
    #: an explicit uncertainty token. Empty means "never escalate".
    cascade_escalation_words: tuple[str, ...] = ("not sure", "uncertain", "cannot", "can't")


class DuplicateFractionSpec(Frozen):
    name: str
    exact: float
    paraphrase: float
    trap: float
    novel: float

    def check_sums_to_one(self) -> None:
        if abs(self.exact + self.paraphrase + self.trap + self.novel - 1.0) > 1e-9:
            raise ValueError(f"workload fraction {self.name} does not sum to 1")


class WorkloadSpec(Frozen):
    request_template: str = 'Explain this statement: "{sentence}"'
    n_per_fraction: int = 500
    seed: int = 20260908
    duplicate_fractions: tuple[DuplicateFractionSpec, ...]

    @model_validator(mode="after")
    def _check_fractions(self) -> WorkloadSpec:
        if self.n_per_fraction <= 0:
            raise ValueError("workload n_per_fraction must be positive")
        for f in self.duplicate_fractions:
            f.check_sums_to_one()
        return self


class JudgeSpec(Frozen):
    rubric: str = ""
    temperature: float = 0.0
    max_tokens: int = 1200
    second_judge_sample: int = 60
    sample_seed: int = 7


class DataPathsSpec(Frozen):
    raw_dir: Path = Path("data/raw")
    cache_dir: Path = Path("data/cache")
    models_dir: Path = Path("data/models")
    runs_dir: Path = Path("data/runs")
    primary_results: Path = Path("results.json")
    archive_dir: Path = Path("archive")


class Config(Frozen):
    gateway: GatewaySpec
    generation: GenerationSpec
    models: ModelsSpec
    prices: PricesSpec
    embeddings: EmbeddingsSpec
    cache: CacheSpec
    router: RouterSpec
    workload: WorkloadSpec
    judge: JudgeSpec
    data: DataPathsSpec

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path} did not load as a YAML mapping")
        cfg = Config.model_validate(raw)
        return cfg
