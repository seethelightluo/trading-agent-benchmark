"""miner_3 2026-07-30 -- Persist cycle-2 screened PASS candidates after full
validation against the ACTIVE library (mom_10d_skip5, vix_beta_cond_60x20,
yield_beta_cond_60x20). Candidates: adx_14, macd_hist_12x26, dxy_beta_cond_60x20.
Writes factors/<factor_id>.json with signal artifact + admission block, then
reloads to verify.
"""
import sys, json, base64, zlib, io, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, "scripts")
from miner_3_20260730_common import load_data, factor_ic_table, coverage_stats, rank_turnover

data = load_data(days=3200)
closes = {a: d["close"].astype(float) for a, d in data.items()}
opens = {a: d["open"].astype(float) for a, d in data.items()}
highs = {a: d["high"].astype(float) for a, d in data.items()}
lows = {a: d["low"].astype(float) for a, d in data.items()}
vols = {a: d["volume"].astype(float).replace(0, np.nan) for a, d in data.items()}
last_vis = max(d.index.max() for d in data.values())


def load_obs(name):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date").sort_index()["close"].astype(float)
    return s[s.index <= last_vis]


VIX = load_obs("VIX")
DXY = load_obs("DXY")


# ---------------- ACTIVE library (exactly as persisted) ----------------
def active_library():
    lib = {}
    for a, c in closes.items():
        lib.setdefault("mom_10d_skip5", {})[a] = c.shift(5) / c.shift(15) - 1.0
        r = c.pct_change()
        vix_r = VIX.pct_change()
        beta_v = r.rolling(60).cov(vix_r) / vix_r.rolling(60).var()
        lib.setdefault("vix_beta_cond_60x20", {})[a] = -beta_v * (VIX / VIX.shift(20) - 1.0)
        us10 = closes["US10Y"]
        yr = us10.pct_change()
        beta_y = r.rolling(60).cov(yr) / yr.rolling(60).var()
        lib.setdefault("yield_beta_cond_60x20", {})[a] = -beta_y * (us10 / us10.shift(20) - 1.0)
    return lib


LIB = active_library()
print("[lib] active factors:", list(LIB.keys()))


def lib_corr(factor):
    fdf = pd.DataFrame(factor).stack()
    fdf = fdf[fdf.notna()]
    out = {}
    for fid, lf in LIB.items():
        ldf = pd.DataFrame(lf).stack()
        both = fdf.index.intersection(ldf.index)
        if len(both) < 100:
            out[fid] = float("nan")
            continue
        rho, _ = pearsonr(fdf.loc[both].values, ldf.loc[both].values)
        out[fid] = float(rho) if np.isfinite(rho) else float("nan")
    vals = [abs(v) for v in out.values() if np.isfinite(v)]
    return (max(vals) if vals else float("nan")), out


# ---------------- candidate factor builders ----------------
def _wild(series, win):
    return series.ewm(alpha=1.0 / win, adjust=False).mean()


def adx_14(c, o, h, l, v, win=14):
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = _wild(tr, win)
    pdi = 100.0 * _wild(plus_dm, win) / atr.replace(0, np.nan)
    mdi = 100.0 * _wild(minus_dm, win) / atr.replace(0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _wild(dx, win)


def macd_hist_12x26(c, o, h, l, v, **kw):
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    return (macd - sig) / c


def dxy_beta_cond_60x20(c, o, h, l, v, **kw):
    dxy = kw["dxy"].reindex(c.index).ffill()
    r = c.pct_change()
    dr = dxy.pct_change()
    beta = r.rolling(60).cov(dr) / dr.rolling(60).var()
    move = dxy / dxy.shift(20) - 1.0
    return -beta * move


CANDIDATES = [
    ("adx_14", "Wilder ADX(14) trend strength", "trend_strength",
     "Wilder ADX(14): smoothed |+DI - -DI| / (+DI + -DI), scaled by 100. Measures trend strength independent of direction.",
     ["close", "high", "low"], {"win": 14}, ["trend", "technical"]),
    ("macd_hist_12x26", "normalized MACD histogram", "macd_hist",
     "Normalized MACD histogram: (EMA12-EMA26) - EMA9 of MACD, divided by close. Trend/momentum balance signal.",
     ["close"], {"fast": 12, "slow": 26, "signal": 9}, ["momentum", "technical"]),
    ("dxy_beta_cond_60x20", "DXY conditional beta", "dxy_beta_cond",
     "-beta(asset_ret, DXY_ret, 60) * (DXY/DXY.shift(20)-1). Conditional USD-regime signal: assets with high beta to the recent USD move get penalized.",
     ["close"], {"beta_win": 60, "dxy_win": 20}, ["cross-asset", "macro", "conditional-beta"]),
]

for fid, name, tag, desc, deps, params, tags in CANDIDATES:
    f = {}
    for a, c in closes.items():
        try:
            if fid == "dxy_beta_cond_60x20":
                f[a] = dxy_beta_cond_60x20(c, opens[a], highs[a], lows[a], vols[a], dxy=DXY)
            elif fid == "adx_14":
                f[a] = adx_14(c, opens[a], highs[a], lows[a], vols[a])
            else:
                f[a] = macd_hist_12x26(c, opens[a], highs[a], lows[a], vols[a])
        except Exception as e:
            print(f"[{fid}] build error: {e}")
            f[a] = pd.Series(dtype=float)
    f = {a: s.replace([np.inf, -np.inf], np.nan) for a, s in f.items()}

    tbl = factor_ic_table(f, data, horizons=(1, 3, 5, 10, 20), min_assets=8, primary_h=10)
    prim = tbl[10]
    if prim is None:
        print(f"[{fid}] DEGENERATE, skip")
        continue
    cov = coverage_stats(f, data)
    to = rank_turnover(f)
    maxrho, rho_map = lib_corr(f)
    gate_ic = abs(prim["ic"]) >= 0.0070
    gate_icir = abs(prim["icir"]) >= 0.0840
    gate_rho = np.isfinite(maxrho) and maxrho < 0.5
    ok = gate_ic and gate_icir and gate_rho
    print(f"[{fid}] ic10={prim['ic']:+.4f} icir10={prim['icir']:+.4f} hit={prim['ic_hit']:.3f} "
          f"n={prim['n_dates']} cov={cov['coverage_asset_days']:.3f} turn={to:.2f} "
          f"maxrho={maxrho:.3f} rho={ {k: round(v,3) for k,v in rho_map.items()} } -> "
          f"{'PASS' if ok else 'fail'}")

    # regime splits
    regimes = {
        "2020-2021": ("2020-01-01", "2021-12-31"),
        "2022-2023": ("2022-01-01", "2023-12-31"),
        "2024-2026": ("2024-01-01", None),
    }
    reg = {}
    for rn, (s, e) in regimes.items():
        t = factor_ic_table(f, data, horizons=(10,), min_assets=8, primary_h=10, start=s, end=e)[10]
        if t:
            reg[rn] = [round(t["ic"], 4), round(t["icir"], 4), t["n_dates"]]
            print(f"    {rn}: IC={t['ic']:+.4f} ICIR={t['icir']:+.4f} n={t['n_dates']}")

    if not ok:
        continue

    metrics = dict(
        ic=prim["ic"], icir=prim["icir"], ic_hit_ratio=prim["ic_hit"],
        n_ic_dates=prim["n_dates"], coverage_asset_days=cov["coverage_asset_days"],
        coverage_dates_ge8=prim["dates_ge8"], turnover_10d_rank=to,
        decay_ic_by_horizon={str(h): (round(v["ic"], 4) if v else None) for h, v in tbl.items()},
        max_abs_library_correlation=round(maxrho, 4),
        library_correlation_detail={k: round(v, 4) for k, v in rho_map.items()},
        regime_ic_icir=reg,
    )
    record = dict(
        factor_id=fid, factor_name=name, version="1.0.0",
        calculation=dict(expression=desc, description=desc),
        dependencies=deps, parameters=params, tags=tags,
        expected_direction=1 if prim["ic"] >= 0 else -1,
        validation=dict(
            status="EFFECTIVE",
            period="2020-01-01..2026-07-29",
            regime_notes="Full-sample 2020-01..2026-07 cross-asset universe (15 instruments); trend-strength/"
                         "conditional-macro family orthogonal to active momentum/VIX/yield factors.",
            metrics=metrics,
        ),
        last_validated="2026-07-30",
        benchmark_admission=dict(
            contract={"ic_threshold": 0.007, "icir_threshold": 0.084, "correlation_threshold": 0.5,
                      "library_capacity": 30, "active_top_k": 10},
            selected_metrics={"ic": prim["ic"], "icir": prim["icir"],
                              "metric_path": "validation.metrics",
                              "reported_max_abs_library_correlation": round(maxrho, 4),
                              "correlation_path": "validation.metrics.max_abs_library_correlation"},
            admitted_at=pd.Timestamp.now().isoformat(),
        ),
    )
    # signal artifact matching existing library format
    fdf = pd.DataFrame(f).sort_index()
    fdf = fdf.reindex(columns=sorted(fdf.columns))
    fdf.to_csv("/tmp/_sig.csv")
    raw = open("/tmp/_sig.csv", "rb").read()
    comp = zlib.compress(raw, 6)
    b64 = base64.b64encode(comp).decode("ascii")
    digest = hashlib.sha256(comp).digest()
    sha = str(int.from_bytes(digest[:8], "big"))
    n_valid = int(np.isfinite(fdf.values).sum())
    record["validation"]["signal_artifact"] = {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {fdf.shape}",
        "columns": list(fdf.columns),
        "shape": list(fdf.shape),
        "n_valid_values": n_valid,
        "sha256": sha,
        "data": b64,
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1)
    with open(path) as fh:
        back = json.load(fh)
    assert back["factor_id"] == fid
    assert back["validation"]["status"] == "EFFECTIVE"
    assert "signal_artifact" in back["validation"]
    assert back["benchmark_admission"]["selected_metrics"]["ic"] == prim["ic"]
    print(f"[persist] wrote {path} status={back['validation']['status']} "
          f"artifact_len={len(b64)} reload_ok=True")

print("DONE")
