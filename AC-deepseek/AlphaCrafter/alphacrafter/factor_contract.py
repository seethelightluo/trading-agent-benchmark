"""Deterministic AlphaCrafter factor-admission and library-capacity contract."""

from __future__ import annotations

import json
import gzip
import zlib
import math
import os
import shutil
import numpy as np
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


def _decode_inline_artifact(artifact: dict) -> np.ndarray | None:
    """Decode an inline signal artifact dict (base64:zlib:csv or base64:zlib:npy)."""
    import base64
    import pandas as pd
    fmt = artifact.get("format", "")
    raw_b64 = artifact.get("data")
    if not raw_b64:
        return None
    try:
        raw = base64.b64decode(raw_b64)
        if "zlib" in fmt:
            raw = zlib.decompress(raw)
        elif "gzip" in fmt:
            raw = gzip.decompress(raw)
        if "csv" in fmt or "csv" in fmt.replace(":", " "):
            import io
            df = pd.read_csv(io.BytesIO(raw), index_col=0)
            return df.to_numpy(dtype=float)
        else:
            import io
            return np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        return None


def _load_signal_artifact(payload: dict, factor_path: Path) -> np.ndarray | None:
    """Load a real signal matrix when the factor provides one.

    Supports three formats:
    1. Inline ``signals`` array (legacy direct).
    2. Inline ``signal_artifact`` dict with ``format`` + ``data`` keys
       (e.g. ``base64:zlib:csv``) -- the standard DeepSeek/Luna miner output.
    3. File-path reference to a ``.npy`` / ``.npz`` / JSON file.

    A summary field such as ``max_abs_library_correlation: 0`` is never a
    signal artifact.  Missing matrices are intentionally returned as None so
    the caller can quarantine the legacy factor instead of weakening rho.
    """
    direct = payload.get("signals")
    artifact = payload.get("signal_artifact")
    validation = payload.get("validation")
    if isinstance(validation, dict):
        direct = direct if direct is not None else validation.get("signals")
        # Check validation.signal_artifact FIRST (standard miner output location)
        artifact = artifact or validation.get("signal_artifact")
        metrics = validation.get("metrics")
        if isinstance(metrics, dict):
            artifact = artifact or metrics.get("signal_artifact")
    if direct is not None:
        try:
            array = np.asarray(direct, dtype=float)
            return array if array.ndim >= 2 else None
        except (TypeError, ValueError):
            return None
    if not artifact:
        return None
    # Case: inline dict artifact with format + data
    if isinstance(artifact, dict):
        return _decode_inline_artifact(artifact)
    # Case: file-path reference
    if not isinstance(artifact, str):
        return None
    path = Path(artifact)
    if not path.is_absolute():
        path = factor_path.parent / path
    if not path.exists():
        return None
    try:
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                array = archive[archive.files[0]]
        else:
            array = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
        return array if array.ndim >= 2 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _pairwise_abs_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute mean cross-sectional absolute Spearman rho over common rows.

    When signal panels have different row counts (e.g. seed factors computed
    on full history vs miner factors on visible window only), align by taking
    the last N rows of each where N = min(left rows, right rows).
    """
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"signal column count differs: {left.shape[1]} vs {right.shape[1]}")
    if left.shape[0] != right.shape[0]:
        n = min(left.shape[0], right.shape[0])
        left = left[-n:]
        right = right[-n:]
    values: list[float] = []
    for row_left, row_right in zip(left, right):
        finite = np.isfinite(row_left) & np.isfinite(row_right)
        if int(finite.sum()) < 2:
            continue
        rank_left = pd_rank(row_left[finite])
        rank_right = pd_rank(row_right[finite])
        if np.std(rank_left) <= 1e-12 or np.std(rank_right) <= 1e-12:
            continue
        values.append(abs(float(np.corrcoef(rank_left, rank_right)[0, 1])))
    if not values:
        raise ValueError("signal artifacts have no common valid cross-section")
    return float(np.mean(values))


def pd_rank(values: np.ndarray) -> np.ndarray:
    """Small dependency-free average-rank implementation."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


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

    # The AC Miner field is provenance/audit metadata, not the conflict gate.
    # The native prompt asks the model to self-report a library correlation,
    # but the benchmark contract must compare both real signal artifacts on a
    # common sample.  Rejecting a candidate here would prevent the lower/higher
    # quality comparison required by the worldline contract.
    reported_correlation, correlation_path = _find_correlation(payload)
    return {
        "ic": selected[0],
        "icir": selected[1],
        "metric_path": selected[2],
        "reported_max_abs_library_correlation": reported_correlation,
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
    admitted: list[tuple[float, str, Path, np.ndarray]] = []
    rejected: list[str] = []
    quarantined: list[str] = []
    evicted: list[str] = []
    conflicts_audit: list[dict[str, Any]] = []

    for path in sorted(root.glob("*.json")):
        # The Screener may persist the active ensemble beside factor files.
        # It is a consumer artifact, not a library member and must never enter
        # admission, capacity sorting, or eviction.
        if path.name == "factor_ensemble.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stamped, metrics = stamp_admission(payload, contract)
            signal = _load_signal_artifact(stamped, path)
            if signal is None:
                raise ValueError(
                    "factor has no recoverable signal artifact; quarantine instead of assuming rho=0"
                )
        except Exception as exc:
            bucket = "quarantine" if "signal artifact" in str(exc) else "rejected"
            archived = _archive(path, bucket)
            atomic_write_json(
                archived.with_suffix(archived.suffix + ".reason.json"),
                {"source": path.name, "reason": str(exc), "contract": contract.as_dict()},
            )
            (quarantined if bucket == "quarantine" else rejected).append(path.name)
            continue
        atomic_write_json(path, stamped)
        admitted.append((float(metrics["quality"]), str(payload.get("factor_id", path.stem)), path, signal))

    # Quality order makes conflict resolution deterministic: when rho >= .5,
    # the first factor wins and the lower-quality member is archived.
    ordered: list[tuple[float, str, Path, np.ndarray]] = []
    for candidate in sorted(admitted, key=lambda item: (-item[0], item[1], item[2].name)):
        conflicts = []
        for kept in ordered:
            rho = _pairwise_abs_spearman(candidate[3], kept[3])
            if rho >= contract.correlation_threshold:
                conflicts.append((kept, rho))
        if conflicts:
            archived = _archive(candidate[2], "evicted")
            conflict_record = {
                "source": candidate[2].name,
                "factor_id": candidate[1],
                "reason": "pairwise correlation conflict; lower quality",
                "quality": candidate[0],
                "conflicts": [
                    {"factor_id": item[1], "abs_spearman_rho": rho}
                    for item, rho in conflicts
                ],
                "contract": contract.as_dict(),
            }
            atomic_write_json(
                archived.with_suffix(archived.suffix + ".reason.json"),
                conflict_record,
            )
            conflicts_audit.append(conflict_record)
            evicted.append(candidate[2].name)
        else:
            ordered.append(candidate)

    # This is the single quality ordering for the library gate.  Reuse the
    # ordered list for both eviction and the audit payload; do not sort the
    # same library a second time just to report it.
    if len(ordered) > contract.library_capacity:
        keep = {path for _, _, path, _ in ordered[: contract.library_capacity]}
        for _, _, path, _ in ordered:
            if path not in keep:
                _archive(path, "evicted")
                evicted.append(path.name)
        ordered = [item for item in ordered if item[2] in keep]

    return {
        "kept": len(ordered),
        "capacity": contract.library_capacity,
        "rejected": rejected,
        "evicted": evicted,
        "kept_files": [path.name for _, _, path, _ in ordered],
        "quarantined": quarantined,
        "conflicts": conflicts_audit,
        "policy": "worldline_pairwise_signal_quality_v1",
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
        if _load_signal_artifact(factor, factor_path) is None:
            raise ValueError(
                f"factor has no recoverable signal artifact: {factor_path.name}"
            )
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
