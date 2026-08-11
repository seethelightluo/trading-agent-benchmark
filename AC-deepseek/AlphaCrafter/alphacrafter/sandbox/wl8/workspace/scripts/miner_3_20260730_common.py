"""Shared validation harness for miner_3 factor research (2026-07-30).
Loads the 15-asset tradable universe through the simulator API (data visible
through 2026-07-29), computes per-asset factor series, and evaluates
cross-sectional rank IC / ICIR / coverage / turnover / decay / library correlation.
One research idea per candidate script; this module only provides infrastructure.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

OBS_ONLY = {"DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"}


def get_watchlist():
    try:
        wl = get_account_dict().get("watch_list") or []
        if wl:
            return list(wl)
    except Exception:
        pass
    return WATCH


def load_data(days=3200):
    """Return dict asset -> DataFrame indexed by date (all OHLCV + fundamentals)."""
    out = {}
    for s in get_watchlist():
        try:
            df = get_stock_daily_data(symbol=s, days=days)
        except Exception:
            df = None
        if df is None or len(df) < 400:
            print(f"[load] {s}: insufficient data ({0 if df is None else len(df)} rows)")
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        out[s] = df
    print(f"[load] loaded {len(out)}/15 instruments; date range "
          f"{min(d.index.min() for d in out.values()).date()} .. {max(d.index.max() for d in out.values()).date()}")
    return out


def align_panel(data, col="close"):
    """Outer-join a column across assets (dates as rows, assets as columns)."""
    return pd.DataFrame({a: d[col].astype(float) for a, d in data.items()})


def factor_ic_table(factor, data, horizons=(1, 2, 3, 5, 10, 20), min_assets=8,
                    primary_h=10, start=None, end=None):
    """factor: dict asset -> pd.Series of factor values (date-indexed).
    Returns per-horizon IC stats and per-date IC list (for primary horizon)."""
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    res = {}
    for h in horizons:
        fwd = {a: c.shift(-h) / c - 1.0 for a, c in closes.items()}
        fdf = pd.DataFrame(factor)
        rdf = pd.DataFrame(fwd)
        common = fdf.index.intersection(rdf.index)
        if start:
            common = common[common >= pd.Timestamp(start)]
        if end:
            common = common[common <= pd.Timestamp(end)]
        ics = []
        n_ge8 = 0
        for d in common:
            f = fdf.loc[d].dropna()
            r = rdf.loc[d].dropna()
            both = f.index.intersection(r.index)
            if len(both) < min_assets:
                continue
            ic, _ = spearmanr(f[both], r[both])
            if np.isfinite(ic):
                ics.append((d, ic))
                if len(both) >= 8:
                    n_ge8 += 1
        if not ics:
            res[h] = None
            continue
        ic_arr = np.array([x[1] for x in ics])
        mean_ic = float(ic_arr.mean())
        std_ic = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else 0.0
        icir = mean_ic / std_ic if std_ic > 0 else 0.0
        hit = float((ic_arr > 0).mean()) if mean_ic >= 0 else float((ic_arr < 0).mean())
        res[h] = dict(n_dates=len(ics), ic=mean_ic, icir=icir, ic_hit=hit,
                      dates_ge8=n_ge8 / len(ics))
    return res


def coverage_stats(factor, data):
    total, valid = 0, 0
    for a, s in factor.items():
        if a not in data:
            continue
        v = s.dropna()
        total += len(s)
        valid += len(v)
    cov = valid / total if total else 0.0
    return dict(coverage_asset_days=cov)


def rank_turnover(factor, step=10, min_assets=8):
    """Mean absolute cross-sectional rank change between observations spaced
    `step` trading days apart (on the common IC-style date axis)."""
    fdf = pd.DataFrame(factor)
    fdf = fdf.dropna(how="all")
    if len(fdf) < 3 * step:
        return float("nan")
    rows = fdf.iloc[::step]
    ranks = rows.rank(axis=1)
    chg = []
    prev = None
    for _, r in ranks.iterrows():
        r = r.dropna()
        if prev is not None:
            both = r.index.intersection(prev.index)
            if len(both) >= min_assets:
                chg.append(float((r[both] - prev[both]).abs().mean()))
        prev = r
    return float(np.mean(chg)) if chg else float("nan")


def library_factors(data):
    """Replicate existing library factors from their persisted definitions."""
    closes = {a: d["close"].astype(float) for a, d in data.items()}
    vix = None
    try:
        vdf = get_stock_daily_data(symbol="VIX", days=3200)
        if vdf is not None:
            vdf = vdf.copy()
            vdf["date"] = pd.to_datetime(vdf["date"])
            vix = vdf.set_index("date").sort_index()["close"].astype(float)
    except Exception:
        pass
    lib = {}
    for a, c in closes.items():
        lib.setdefault("mom_10d_skip5", {})[a] = c.shift(5) / c.shift(15) - 1.0
        lib.setdefault("mom_120d_skip5", {})[a] = c.shift(5) / c.shift(125) - 1.0
        lib.setdefault("vol_of_vol20x60", {})[a] = c.pct_change().rolling(20).std().rolling(60).std()
        if vix is not None:
            r = c.pct_change()
            beta = r.rolling(60).cov(vix.pct_change()) / vix.pct_change().rolling(60).var()
            vix_move = vix / vix.shift(20) - 1.0
            lib.setdefault("vix_beta_cond_60x20", {})[a] = -beta * vix_move
    return lib


def max_library_corr(factor, data):
    """Pearson rho between the candidate's long panel (date,asset) values and
    each library factor's panel; report max |rho|."""
    fdf = pd.DataFrame(factor).stack()
    fdf = fdf[fdf.notna()]
    if len(fdf) < 100:
        return float("nan"), {}
    out = {}
    for fid, lf in library_factors(data).items():
        ldf = pd.DataFrame(lf).stack()
        both = fdf.index.intersection(ldf.index)
        if len(both) < 100:
            out[fid] = float("nan")
            continue
        rho, _ = pearsonr(fdf.loc[both].values, ldf.loc[both].values)
        out[fid] = float(rho)
    vals = [abs(v) for v in out.values() if np.isfinite(v)]
    return (max(vals) if vals else float("nan")), out


def run_validation(factor_id, factor_name, expression, description, deps, params,
                   factor_series, data, tags, regime_notes, min_assets=8):
    """Full validation + printing; returns metrics dict (or None if degenerate)."""
    print("=" * 70)
    print(f"FACTOR {factor_id}: {factor_name}")
    print(f"  expression: {expression}")
    print(f"  instruments with valid values: {sum(1 for a, s in factor_series.items() if s.dropna().shape[0] > 100)}/15")

    tbl = factor_ic_table(factor_series, data, min_assets=min_assets)
    if tbl[10] is None:
        print("  !! degenerate: no valid IC dates at primary horizon")
        return None
    prim = tbl[10]
    gate_ic = abs(prim["ic"]) >= 0.0070
    gate_icir = abs(prim["icir"]) >= 0.0840
    print(f"  primary horizon=10: IC={prim['ic']:.4f} ICIR={prim['icir']:.4f} "
          f"hit={prim['ic_hit']:.3f} n_dates={prim['n_dates']} "
          f"dates_ge8={prim['dates_ge8']:.3f} | gate IC>={0.0070} gate ICIR>={0.0840} -> "
          f"{'PASS' if (gate_ic and gate_icir) else 'FAIL'}")
    print("  decay by horizon:", {str(h): (round(v['ic'], 4) if v else None) for h, v in tbl.items()})

    cov = coverage_stats(factor_series, data)
    to = rank_turnover(factor_series)
    print(f"  coverage_asset_days={cov['coverage_asset_days']:.3f} turnover_10d_rank={to:.3f}")

    # regime splits
    regime_splits = {
        "2020-2021 (COVID/recovery)": ("2020-01-01", "2021-12-31"),
        "2022-2023 (tightening/AI)": ("2022-01-01", "2023-12-31"),
        "2024-2026-07 (crypto/commodity)": ("2024-01-01", None),
    }
    reg = {}
    for name, (s, e) in regime_splits.items():
        t = factor_ic_table(factor_series, data, horizons=(10,), min_assets=min_assets,
                            primary_h=10, start=s, end=e)[10]
        if t:
            reg[name] = (t["ic"], t["icir"], t["n_dates"])
            print(f"    regime {name}: IC={t['ic']:.4f} ICIR={t['icir']:.4f} n={t['n_dates']}")

    maxrho, rho_map = max_library_corr(factor_series, data)
    print(f"  library corr: { {k: round(v, 3) for k, v in rho_map.items()} } max_abs={maxrho:.3f}")

    metrics = dict(
        ic=prim["ic"], icir=prim["icir"], ic_hit_ratio=prim["ic_hit"],
        n_ic_dates=prim["n_dates"], coverage_asset_days=cov["coverage_asset_days"],
        coverage_dates_ge8=prim["dates_ge8"], turnover_10d_rank=to,
        decay_ic_by_horizon={str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()},
        max_abs_library_correlation=(round(maxrho, 4) if np.isfinite(maxrho) else None),
        library_correlation_detail={k: round(v, 4) for k, v in rho_map.items()},
        regime_ic_icir={k: [round(v[0], 4), round(v[1], 4), v[2]] for k, v in reg.items()},
    )
    return metrics, dict(
        factor_id=factor_id, factor_name=factor_name,
        calculation=dict(expression=expression, description=description),
        dependencies=deps, parameters=params, tags=tags,
        validation=dict(
            status="EFFECTIVE" if (gate_ic and gate_icir) else "REJECTED",
            period="2020-01-01..2026-07-29",
            regime_notes=regime_notes,
            metrics=metrics,
        ),
        last_validated="2026-07-30",
    )
