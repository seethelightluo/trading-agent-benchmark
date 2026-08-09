"""Deterministic AlphaCrafter factor-admission and library-capacity contract."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from alphacrafter.utils.atomic_io import atomic_write_json


@dataclass(frozen=True)
class FactorContract:
    ic_threshold: float = 0.007
    icir_threshold: float = 0.084
    correlation_threshold: float = 0.5
    library_capacity: int = 30
    active_top_k: int = 10

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None = None) -> "FactorContract":
        values = values or {}
        return cls(
            ic_threshold=float(os.environ.get(
                "AC_FACTOR_IC_THRESHOLD", values.get("ic_threshold", cls.ic_threshold)
            )),
            icir_threshold=float(os.environ.get(
                "AC_FACTOR_ICIR_THRESHOLD", values.get("icir_threshold", cls.icir_threshold)
            )),
            correlation_threshold=float(os.environ.get(
                "AC_FACTOR_CORRELATION_THRESHOLD",
                values.get("correlation_threshold", cls.correlation_threshold),
            )),
            library_capacity=int(os.environ.get(
                "AC_FACTOR_LIBRARY_CAPACITY",
                values.get("library_capacity", cls.library_capacity),
            )),
            active_top_k=int(os.environ.get(
                "AC_FACTOR_ACTIVE_TOP_K", values.get("active_top_k", cls.active_top_k)
            )),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ic_threshold": self.ic_threshold,
            "icir_threshold": self.icir_threshold,
            "correlation_threshold": self.correlation_threshold,
            "library_capacity": self.library_capacity,
            "active_top_k": self.active_top_k,
        }


_IC_KEYS = (
    "ic", "IC", "ic_mean", "mean_ic", "daily_ic", "daily_paper_ic",
    "daily_paper_IC", "mean_daily_paper_ic", "daily_paper_ic_abs",
    "ic_mean_daily", "ic_abs_daily", "mean_daily_ic", "mean_ic_daily",
    "ic_paper_mean",
)
_ICIR_KEYS = (
    "icir", "ICIR", "ic_ir", "daily_icir", "daily_paper_icir",
    "daily_paper_ICIR", "icir_daily", "paper_icir", "ic_paper_icir",
)
_CORR_KEYS = (
    "max_abs_library_correlation",
    "max_library_correlation",
    "max_abs_correlation",
    "max_correlation",
)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _walk_dicts(
    value: Any, path: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], dict]]:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _walk_dicts(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_dicts(child, path + (str(index),))


def _metric_pairs(payload: dict) -> list[tuple[float, float, str]]:
    pairs: list[tuple[float, float, str]] = []
    for path, mapping in _walk_dicts(payload):
        ic = next((_finite_float(mapping.get(key)) for key in _IC_KEYS if key in mapping), None)
        icir = next((_finite_float(mapping.get(key)) for key in _ICIR_KEYS if key in mapping), None)
        if ic is not None and icir is not None:
            pairs.append((ic, icir, ".".join(path) or "root"))
    return pairs


def _find_correlation(payload: dict) -> tuple[float | None, str | None]:
    for path, mapping in _walk_dicts(payload):
        for key in _CORR_KEYS:
            if key in mapping:
                value = _finite_float(mapping[key])
                if value is not None:
                    return abs(value), ".".join(path + (key,))
    return None, None


def evaluate_factor(payload: dict, contract: FactorContract) -> dict[str, Any]:
    validation = payload.get("validation") if isinstance(payload, dict) else None
    status = str((validation or {}).get("status", "")).upper()
    if status not in {"EFFECTIVE", "ACTIVE"}:
        raise ValueError("factor validation.status must be EFFECTIVE or ACTIVE")

    passing = [
        pair for pair in _metric_pairs(payload)
        if abs(pair[0]) >= contract.ic_threshold and abs(pair[1]) >= contract.icir_threshold
    ]
    if not passing:
        raise ValueError(
            "factor has no same-horizon IC/ICIR pair passing "
            f"|IC|>={contract.ic_threshold:.4f} and |ICIR|>={contract.icir_threshold:.4f}"
        )
    selected = max(passing, key=lambda pair: abs(pair[0]) * abs(pair[1]))

    correlation, correlation_path = _find_correlation(payload)
    if correlation is None:
        raise ValueError(
            "factor must report validation.metrics.max_abs_library_correlation "
            "(use 0.0 for the first admitted factor)"
        )
    if correlation >= contract.correlation_threshold:
        raise ValueError(
            f"factor correlation {correlation:.4f} must be < {contract.correlation_threshold:.4f}"
        )
    return {
        "ic": selected[0],
        "icir": selected[1],
        "metric_path": selected[2],
        "max_abs_library_correlation": correlation,
        "correlation_path": correlation_path,
        "quality": abs(selected[0]) * abs(selected[1]),
    }


def stamp_admission(payload: dict, contract: FactorContract) -> tuple[dict, dict]:
    result = evaluate_factor(payload, contract)
    stamped = dict(payload)
    stamped["benchmark_admission"] = {
        "contract": contract.as_dict(),
        "selected_metrics": result,
        "admitted_at": datetime.now().isoformat(),
    }
    return stamped, result


def _with_first_admission_correlation(payload: dict) -> dict:
    """Materialize the FM convention for the first library member.

    AlphaCrafter factors are persisted as definitions and summary metrics, not
    as the signal matrices used by FactorMiner's in-memory correlation check.
    The first admitted factor has no predecessor, so FM treats its maximum
    library correlation as zero.  Make that convention explicit in the
    persisted JSON instead of silently weakening the rule for later factors.
    """
    materialized = json.loads(json.dumps(payload))
    validation = materialized.setdefault("validation", {})
    if not isinstance(validation, dict):
        validation = {}
        materialized["validation"] = validation
    metrics = validation.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        validation["metrics"] = metrics
    metrics["max_abs_library_correlation"] = 0.0
    return materialized


def _archive(path: Path, bucket: str) -> Path:
    target_dir = path.parent / bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    if target.exists():
        target = target_dir / (
            f"{path.stem}.{datetime.now().strftime('%Y%m%dT%H%M%S%f')}{path.suffix}"
        )
    shutil.move(str(path), str(target))
    return target


def enforce_library(directory: str | Path, contract: FactorContract) -> dict[str, Any]:
    """Validate active factor files and retain all up to cap, then best-N above cap."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    admitted: list[tuple[float, Path]] = []
    rejected: list[str] = []
    inferred_first_correlation: list[str] = []

    # A factor already stamped by this gate means the rolling library is not
    # empty.  Missing correlation is only recoverable for the first admitted
    # member; after that, new factors must report their FM-compatible
    # max_abs_library_correlation explicitly.
    existing_admitted = False
    for existing in root.glob("*.json"):
        if existing.name == "factor_ensemble.json":
            continue
        try:
            existing_payload = json.loads(existing.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(existing_payload, dict) and existing_payload.get("benchmark_admission"):
            existing_admitted = True
            break

    for path in sorted(root.glob("*.json")):
        # The Screener may persist the active ensemble beside factor files.
        # It is a consumer artifact, not a library member and must never enter
        # admission, capacity sorting, or eviction.
        if path.name == "factor_ensemble.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            try:
                stamped, metrics = stamp_admission(payload, contract)
            except ValueError as exc:
                if (
                    not existing_admitted
                    and not admitted
                    and "must report validation.metrics.max_abs_library_correlation" in str(exc)
                ):
                    payload = _with_first_admission_correlation(payload)
                    stamped, metrics = stamp_admission(payload, contract)
                    metrics["correlation_inferred_for_first_admission"] = True
                    inferred_first_correlation.append(path.name)
                else:
                    raise
        except Exception as exc:
            archived = _archive(path, "rejected")
            atomic_write_json(
                archived.with_suffix(archived.suffix + ".reason.json"),
                {"source": path.name, "reason": str(exc), "contract": contract.as_dict()},
            )
            rejected.append(path.name)
            continue
        atomic_write_json(path, stamped)
        admitted.append((float(metrics["quality"]), path))

    # This is the single quality ordering for the library gate.  Reuse the
    # ordered list for both eviction and the audit payload; do not sort the
    # same library a second time just to report it.
    ordered = sorted(
        admitted, key=lambda item: (item[0], item[1].name), reverse=True
    )
    evicted: list[str] = []
    if len(ordered) > contract.library_capacity:
        keep = {path for _, path in ordered[: contract.library_capacity]}
        for _, path in admitted:
            if path not in keep:
                _archive(path, "evicted")
                evicted.append(path.name)
        ordered = [(score, path) for score, path in ordered if path in keep]

    return {
        "kept": len(ordered),
        "capacity": contract.library_capacity,
        "rejected": rejected,
        "evicted": evicted,
        "kept_files": [path.name for _, path in ordered],
        "inferred_first_correlation": inferred_first_correlation,
    }


def validate_ensemble_payload(
    payload: dict[str, Any],
    library_directory: str | Path,
    contract: FactorContract,
) -> dict[str, Any]:
    selected = payload.get("selected_factors")
    if not isinstance(selected, list):
        raise ValueError("factor_ensemble.json must contain selected_factors as a list")
    if len(selected) > contract.active_top_k:
        raise ValueError(
            f"active factor count {len(selected)} exceeds top-k {contract.active_top_k}"
        )

    library_ids: set[str] = set()
    for factor_path in Path(library_directory).glob("*.json"):
        if factor_path.name == "factor_ensemble.json":
            continue
        factor = json.loads(factor_path.read_text(encoding="utf-8"))
        evaluate_factor(factor, contract)
        if factor.get("factor_id"):
            library_ids.add(str(factor["factor_id"]))

    selected_ids: list[str] = []
    weights: list[float] = []
    for item in selected:
        if not isinstance(item, dict) or not item.get("factor_id"):
            raise ValueError("every selected factor must contain factor_id")
        factor_id = str(item["factor_id"])
        if factor_id not in library_ids:
            raise ValueError(f"selected factor is not in the admitted library: {factor_id}")
        selected_ids.append(factor_id)
        if "weight" in item:
            weight = _finite_float(item["weight"])
            if weight is None or weight < 0:
                raise ValueError(f"invalid ensemble weight for {factor_id}")
            weights.append(weight)

    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("factor ensemble contains duplicate factor_id values")
    if weights and (len(weights) != len(selected) or abs(sum(weights) - 1.0) > 1e-6):
        raise ValueError("explicit factor ensemble weights must be non-negative and sum to 1")
    return {"selected_count": len(selected), "selected_ids": selected_ids}


def validate_ensemble(
    ensemble_path: str | Path,
    library_directory: str | Path,
    contract: FactorContract,
) -> dict[str, Any]:
    payload = json.loads(Path(ensemble_path).read_text(encoding="utf-8"))
    return validate_ensemble_payload(payload, library_directory, contract)
