"""miner_2 2026-07-30 cycle 21: explore & validate NOVEL close-only factors.

Lessons from prior cycles:
  - The post-miner gate evaluates factors in a STRICT namespace:
      eval(expr, {'__builtins__':{}}, {'pd':pd, 'np':np, 'close':panel})
    so only close/pd/np-based expressions are recoverable (volume/VIX/DXY factors
    were quarantined before).  All candidates here are strict-namespace safe.
  - The gate recomputes pairwise rho from REAL signal artifacts (npy files), so
    we persist a matrix file per factor; all matrices share the exact same grid
    (union calendar, 15 assets in ASSETS order).
  - Admission gates: |IC10| >= 0.007 and |ICIR10| >= 0.084 (h=10 daily rank IC,
    >=8 valid assets per date).

This script only SCREENS + VALIDATES; persistence happens in a follow-up step.
"""
import sys, json
import numpy as np
import pandas as pd

VISIBLE = "2026-07-29"
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
HORIZONS = [1, 2, 3, 5, 10, 20]
ADM_H = 10
MIN_ASSETS = 8


def load_close(asset):
    df = pd.read_csv(f'../persistent/stock_data/{asset}.csv', parse_dates=['date'])
    df = df[df['date'] <= pd.Timestamp(VISIBLE)].set_index('date')['close'].astype(float)
    return df


def build_panel():
    return pd.DataFrame({a: load_close(a) for a in ASSETS})


def forward_returns(prices, horizon):
    fwd = {}
    for a in prices.columns:
        s = prices[a].dropna()
        fwd[a] = (s.shift(-horizon) / s - 1.0)
    return pd.DataFrame(fwd).reindex(prices.index)


def spearman_ic(factor_df, fwd_df):
    dates, ics = [], []
    idx = factor_df.index.intersection(fwd_df.index)
    for d in idx:
        f = factor_df.loc[d]
        r = fwd_df.loc[d]
        mask = f.notna() & r.notna()
        if mask.sum() < MIN_ASSETS:
            continue
        ic = f[mask].corr(r[mask], method='spearman')
        if np.isfinite(ic):
            dates.append(d)
            ics.append(ic)
    return pd.Series(ics, index=dates)


def mean_rank_turnover(factor_df, step=10):
    ranks = factor_df.rank(axis=1, pct=True)
    chg = (ranks - ranks.shift(step)).abs()
    return float(chg.stack().mean())


panel = build_panel()
print(f"panel shape={panel.shape} dates={panel.index.min().date()}..{panel.index.max().date()}")
print(f"assets={list(panel.columns)}")

env = {"pd": pd, "np": np, "close": panel}
RET = "close.pct_change()"

# ---------------------------------------------------------------------------
# Novel candidates (strict-namespace recoverable, not tried in prior cycles)
# ---------------------------------------------------------------------------
EXPRS = {
    # ---- drift / risk-adjusted trend ----
    "tstat_20":        f"{RET}.rolling(20, min_periods=10).mean() / ({RET}.rolling(20, min_periods=10).std() + 1e-12)",
    "tstat_60":        f"{RET}.rolling(60, min_periods=30).mean() / ({RET}.rolling(60, min_periods=30).std() + 1e-12)",
    "mom_demean_20":   "(close.shift(5)/close.shift(25)-1.0).sub((close.shift(5)/close.shift(25)-1.0).mean(axis=1), axis=0)",
    "mom_demean_60":   "(close.shift(5)/close.shift(65)-1.0).sub((close.shift(5)/close.shift(65)-1.0).mean(axis=1), axis=0)",
    # ---- trend location / acceleration ----
    "trend_accel_20x60": "(close.rolling(20, min_periods=10).mean()/close.rolling(60, min_periods=30).mean()).pct_change(20)",
    "dist_high_252":   "(close/close.rolling(252, min_periods=60).max() - 1.0)",
    "range_pos_60":    "(close - close.rolling(60, min_periods=30).min()) / (close.rolling(60, min_periods=30).max() - close.rolling(60, min_periods=30).min() + 1e-12) - 0.5",
    # ---- volatility regime ----
    "bb_width_20":     f"4.0*{RET}.rolling(20, min_periods=10).std() / close.rolling(20, min_periods=10).mean()",
    "vol_ratio_60x252": f"{RET}.rolling(60, min_periods=30).std() / ({RET}.rolling(252, min_periods=60).std() + 1e-12)",
    "vol_trend_20x60": f"{RET}.rolling(20, min_periods=10).std() / ({RET}.rolling(60, min_periods=30).std() + 1e-12)",
    # ---- path shape / asymmetry ----
    "updown_asym_10":  f"({RET}.clip(lower=0).rolling(10, min_periods=5).sum() - {RET}.clip(upper=0).rolling(10, min_periods=5).sum().abs()) / ({RET}.abs().rolling(10, min_periods=5).sum() + 1e-12)",
    "autocorr_20x1":   f"{RET}.rolling(20, min_periods=10).corr({RET}.shift(1))",
    "skew_60":         f"{RET}.rolling(60, min_periods=30).skew()",
    "kurt_60":         f"{RET}.rolling(60, min_periods=30).kurt()",
    # ---- momentum consistency / efficiency ----
    "mom_consistency_120": f"(({RET}.rolling(20, min_periods=10).mean() > 0).rolling(6, min_periods=3).mean())",
    "eff_ratio_120":   "(close - close.shift(120)).abs() / close.diff().abs().rolling(120, min_periods=60).sum()",
}

fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}
fwd_adm = fwd_cache[str(ADM_H)]

results = {}
signals = {}
print("\n=== strict-namespace eval + validation ===")
for fid, exp in EXPRS.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        if isinstance(sig, pd.Series):
            sig = sig.to_frame()
        sig = sig.reindex(index=panel.index, columns=panel.columns)
    except Exception as e:
        print(f"  {fid:20s} EVAL FAIL ({type(e).__name__}): {str(e)[:60]}")
        continue
    if not isinstance(sig, pd.DataFrame) or sig.shape != panel.shape:
        print(f"  {fid:20s} BAD SHAPE {getattr(sig, 'shape', None)}")
        continue
    signals[fid] = sig
    ics = spearman_ic(sig, fwd_adm)
    if len(ics) < 30:
        print(f"  {fid:20s} n_ic={len(ics)} TOO FEW")
        results[fid] = dict(ic=0.0, icir=0.0, n=len(ics), gate=False)
        continue
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
    hit = float((ics > 0).mean()) if ic >= 0 else float((ics < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, fwd_cache[str(h)]).mean()), 4) for h in HORIZONS}
    cov = float(sig.notna().sum().sum()) / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    regime = {}
    for b0, b1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = ics[(ics.index >= b0) & (ics.index <= b1)]
        if len(sub) >= 30:
            sd = sub.std()
            regime[f"{b0[:4]}-{b1[:4]}"] = {"ic": round(float(sub.mean()), 4),
                                            "icir": round(float(sub.mean() / sd), 4) if sd > 0 else 0.0,
                                            "n_dates": int(len(sub))}
    results[fid] = dict(ic=round(ic, 4), icir=round(icir, 4), hit=round(hit, 3),
                        n=int(len(ics)), cov=round(cov, 4), dates_ge8=round(n_ge8 / len(sig), 4),
                        turn=round(turn, 4), decay=decay, regime=regime,
                        gate=gate, quality=round(abs(ic) * abs(icir), 6))
    print(f"  {fid:20s} n={len(ics):5d} ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} "
          f"cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no '} q={results[fid]['quality']:.5f}")

passers = [fid for fid, r in results.items() if r["gate"]]
print(f"\nPASSERS: {passers} ({len(passers)}/{len(results)})")

# pairwise |rho| among passers (pooled Pearson, gate-style from real signal)
print("\n=== pairwise |rho| among passers (pooled) ===")
rho = pd.DataFrame(index=passers, columns=passers, dtype=float)
for i, a in enumerate(passers):
    for j, b in enumerate(passers):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        r = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[a, b] = rho.loc[b, a] = r
for a in passers:
    print(f"  {a:20s} " + " ".join(f"{b[:6]}={rho.loc[a,b]:+.2f}" for b in passers if b != a and pd.notna(rho.loc[a, b])))

# greedy admission: quality-ranked, max |rho| < 0.5 vs admitted
admitted = []
for fid in sorted(passers, key=lambda f: -results[f]["quality"]):
    if all(abs(rho.loc[fid, b]) < 0.5 for b in admitted):
        admitted.append(fid)
print(f"\nADMITTED (greedy, rho<0.5): {admitted}")

json.dump({"panel_shape": list(panel.shape),
           "results": results, "passers": passers, "admitted": admitted,
           "pairwise_rho": {a: {b: (None if pd.isna(rho.loc[a, b]) else round(float(rho.loc[a, b]), 4))
                                for b in passers} for a in passers}},
          open("scripts/_miner2_cycle21_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner2_cycle21_results.json")
