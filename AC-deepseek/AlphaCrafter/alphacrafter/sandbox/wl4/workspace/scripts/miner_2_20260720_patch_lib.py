"""Append per-asset-calendar helpers to the shared research lib (additive)."""
from pathlib import Path

p = Path('scripts/factor_research_lib.py')
txt = p.read_text()
add = '''

# ---------------------------------------------------------------------------
# Per-asset-calendar helpers (fix union-calendar NaN bug: BTC/ETH trade 7d,
# equities trade 5d, so rolling ops on the outer-joined panel break).
# Each asset's factor/forward-return is computed on its OWN trading calendar.
# ---------------------------------------------------------------------------

def forward_returns_pa(closes: pd.DataFrame, horizon: int):
    """Per-asset forward h-day return computed on each asset's own calendar."""
    out = {}
    for a in closes.columns:
        s = closes[a].dropna()
        if len(s) <= horizon:
            continue
        out[a] = (s.shift(-horizon) / s - 1.0)
    return pd.DataFrame(out).sort_index()


def apply_per_asset(closes: pd.DataFrame, func, assets=None):
    """Apply func(series)->series to each asset's own dropna'd close series,
    return outer-joined DataFrame on the union index (per-asset values present
    only on that asset's own trading dates)."""
    assets = assets if assets is not None else list(closes.columns)
    out = {}
    for a in assets:
        if a not in closes.columns:
            continue
        s = closes[a].dropna()
        if len(s) < 30:
            continue
        try:
            out[a] = func(s).astype(float)
        except Exception as exc:  # pragma: no cover
            print(f"apply_per_asset: {a} failed: {exc}")
    return pd.DataFrame(out).sort_index()


def rank_ic_series_pa(factor_panel: pd.DataFrame, closes: pd.DataFrame,
                      horizon: int, min_valid: int = 8):
    """Per-asset-calendar rank IC: forward returns computed per asset, so
    equities use equity trading days and crypto uses crypto days."""
    fwd = forward_returns_pa(closes, horizon)
    return rank_ic_series(factor_panel, fwd, min_valid)


def decay_profile_pa(factor_panel, closes, horizons=(1, 2, 3, 5, 10, 20),
                     min_valid=8, expected_sign=1):
    out = {}
    for h in horizons:
        ics = rank_ic_series_pa(factor_panel, closes, h, min_valid)
        if len(ics):
            out[str(h)] = round(float(ics.mean()), 4)
    return out


def full_eval_pa(factor_panel, closes, horizons=(1, 2, 3, 5, 10, 20),
                 min_valid=8, expected_sign=1, library=None, admission_horizon=10):
    """Per-asset-calendar full validation at the admission horizon."""
    ics = rank_ic_series_pa(factor_panel, closes, admission_horizon, min_valid)
    m = summarize_ic(ics, expected_sign)
    m.update(coverage_metrics(factor_panel, min_valid=min_valid))
    m["turnover_10d_rank"] = turnover_rank(factor_panel, admission_horizon)
    m["decay_ic_by_horizon"] = decay_profile_pa(factor_panel, closes, horizons,
                                                min_valid, expected_sign)
    if library is not None:
        corr, key = max_library_corr(factor_panel, library)
        m["max_abs_library_correlation"] = corr
        m["max_corr_factor"] = key
    return m, ics


def load_library_artifacts(valid_until=None):
    """Decode persisted signal artifacts (base64:zlib:csv) from factors/*.json
    into {factor_id: DataFrame}, restricted to dates <= valid_until."""
    import base64 as _b64
    import zlib as _zlib
    import io as _io
    out = {}
    if not LIB_DIR.exists():
        return out
    for p in sorted(LIB_DIR.glob("*.json")):
        if p.name.endswith(".bak"):
            continue
        try:
            d = json.loads(p.read_text())
            sa = d.get("validation", {}).get("signal_artifact")
            if not sa or sa.get("format") != "base64:zlib:csv":
                continue
            csv = _zlib.decompress(_b64.b64decode(sa["data"])).decode("utf-8")
            df = pd.read_csv(_io.StringIO(csv), index_col=0)
            df.index = pd.to_datetime(df.index)
            if valid_until is not None:
                df = df.loc[df.index <= valid_until]
            out[d["factor_id"]] = df
        except Exception as exc:
            print(f"load_library_artifacts: {p.name} failed: {exc}")
    return out
'''
p.write_text(txt + add)
print("appended OK")
