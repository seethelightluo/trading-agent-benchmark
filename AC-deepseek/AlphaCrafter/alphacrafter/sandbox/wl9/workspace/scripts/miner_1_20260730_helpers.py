"""Shared helpers for factor mining: data loading, IC analysis, persistence.

Restricted to data visible through the sim's last completed trading day
(2026-07-29) to avoid lookahead. Validation window: 2020-01-01..2026-07-29.

UPDATE 2026-07-30 cycle: the mixed-calendar cross-asset universe (BTC/ETH trade
7d/wk; equity/commodity/yield series trade on their own calendars) makes
union-calendar rolling computations NaN-contaminated. All forward-return and
decay computations are therefore performed per asset on each asset's own
calendar, then reindexed to the union date axis only for cross-sectional
alignment.
"""
import json
import zlib
import base64
import io
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MAX_DATE = "2026-07-29"
MIN_DATE = "2020-01-01"


def load_close(symbol, root="../persistent"):
    for sub, name in (("stock_data", symbol), ("index_data", symbol)):
        p = Path(root) / sub / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            df = df[df["date"] <= MAX_DATE].set_index("date")
            return df
    return None


def load_panel(symbols, root="../persistent"):
    closes = {}
    for s in symbols:
        df = load_close(s, root)
        if df is not None and "close" in df:
            closes[s] = df["close"].astype(float)
    panel = pd.DataFrame(closes).dropna(how="all").sort_index()
    panel = panel[panel.index >= MIN_DATE]
    return panel


def forward_returns(ret_panel, horizon):
    """Per-asset h-day forward cumulative return aligned to ret_panel index.

    Each asset's own calendar is used (s.shift(-h)/s - 1), then reindexed to
    the union index. This avoids NaN contamination from mixed calendars.
    """
    out = {}
    for a in ret_panel.columns:
        s = ret_panel[a].dropna()
        out[a] = (s.shift(-horizon) / s - 1.0).reindex(ret_panel.index)
    return pd.DataFrame(out)


def factor_ic_report(panel, forward_ret, min_valid=8, horizon=10):
    """Daily cross-sectional Spearman IC between factor panel and forward returns."""
    dates = panel.index
    ics, n_valid, hit = [], [], 0
    for t in dates:
        f = panel.loc[t]
        r = forward_ret.loc[t]
        mask = f.notna() & r.notna()
        n = int(mask.sum())
        if n < min_valid:
            continue
        ic = f[mask].rank().corr(r[mask].rank())
        if not np.isfinite(ic):
            continue
        ics.append(ic)
        n_valid.append(n)
    ics = np.array(ics)
    if len(ics) == 0:
        return None
    mean_ic = float(ics.mean())
    std_ic = float(ics.std(ddof=1)) if len(ics) > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    hit_ratio = float((np.sign(ics) == np.sign(mean_ic)).mean()) if mean_ic != 0 else 0.5
    return {
        "horizon": horizon,
        "ic": mean_ic,
        "icir": icir,
        "ic_hit_ratio": hit_ratio,
        "n_ic_dates": len(ics),
        "mean_n_valid": float(np.mean(n_valid)),
    }


def factor_turnover(panel):
    """Average cross-sectional rank change between consecutive dates (0..1 scale)."""
    ranks = panel.rank(axis=1)
    chg = ranks.diff().abs().mean(axis=1)
    return float(chg.mean())


def coverage(panel, n_assets=15):
    valid = int(panel.notna().sum().sum())
    total = int(panel.shape[0] * n_assets)
    dates_ge8 = float((panel.notna().sum(axis=1) >= 8).mean())
    return {"coverage_asset_days": valid / total if total else 0.0,
            "coverage_dates_ge8": dates_ge8}


def decay_report(factor_panel, ret_panel, horizons=(1, 2, 3, 5, 10, 20)):
    out = {}
    for h in horizons:
        fwd = forward_returns(ret_panel, h)
        r = factor_ic_report(factor_panel, fwd, horizon=h)
        out[str(h)] = round(r["ic"], 4) if r else None
    return out


def panel_correlation(a, b):
    """Pearson correlation of two factor panels on common (date, asset) pairs."""
    common = a.index.intersection(b.index)
    ac = a.loc[common].stack()
    bc = b.loc[common].stack()
    m = ac.notna() & bc.notna()
    if m.sum() < 30:
        return 0.0
    return float(ac[m].corr(bc[m]))


def max_library_correlation(candidate, library_dir="factors"):
    """Max |Pearson rho| between candidate signal panel and each library factor artifact."""
    arts = []
    for p in sorted(Path(library_dir).glob("*.json")):
        if p.name == "factor_ensemble.json":
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            art = d.get("validation", {}).get("signal_artifact")
            if not art:
                continue
            data = base64.b64decode(art["data"])
            raw = zlib.decompress(data).decode()
            df = pd.read_csv(io.StringIO(raw), index_col=0)
            df.index = pd.to_datetime(df.index)
            arts.append((d["factor_id"], df))
        except Exception as e:
            print(f"  [skip artifact {p.name}: {e}]")
    if not arts:
        return 0.0
    rho = {fid: panel_correlation(candidate, df) for fid, df in arts}
    return max(abs(v) for v in rho.values()), rho


def persist_factor(meta, signal_panel, out_path):
    """Encode signal panel as base64:zlib:csv and write the JSON record."""
    buf = io.StringIO()
    signal_panel.round(10).to_csv(buf)
    raw = buf.getvalue().encode()
    blob = base64.b64encode(zlib.compress(raw, 9)).decode()
    sha = hashlib.sha256(blob.encode()).hexdigest()[:16]
    art = {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {signal_panel.shape}",
        "columns": list(signal_panel.columns),
        "shape": list(signal_panel.shape),
        "n_valid_values": int(signal_panel.notna().sum().sum()),
        "sha256": sha,
        "data": blob,
    }
    meta.setdefault("validation", {})["signal_artifact"] = art
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=1)
    return out_path


def verify_factor(path, factor_id):
    with open(path) as f:
        d = json.load(f)
    ok = (d.get("factor_id") == factor_id
          and d.get("validation", {}).get("status") == "EFFECTIVE"
          and d.get("validation", {}).get("metrics", {}).get("ic") is not None
          and "signal_artifact" in d.get("validation", {}))
    print(f"verify {path}: id_ok={d.get('factor_id')==factor_id} "
          f"status={d.get('validation',{}).get('status')} "
          f"artifact={'yes' if 'signal_artifact' in d.get('validation',{}) else 'NO'}")
    return ok