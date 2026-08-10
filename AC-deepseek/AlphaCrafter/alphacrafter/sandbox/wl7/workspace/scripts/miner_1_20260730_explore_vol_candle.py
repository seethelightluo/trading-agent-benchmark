"""
miner_1 exploration: volume/liquidity + candle/range factor families.
Universe: 15 tradable cross-asset instruments. Validation window 2020-01-01..2026-07-15
(factor date window), data visible only through 2026-07-29 (no look-ahead).
Admission gates (benchmark-wide): |IC|>=0.007, |ICIR|>=0.084 at horizon 10.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import glob, os, json, sys

WATCH = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MAX_VISIBLE = "2026-07-29"     # last visible trading day (research cut)
FACTOR_LAST = "2026-07-15"     # last factor date used for IC (research window end)
MIN_ASSETS = 8

def load_panel():
    frames = {}
    for s in WATCH:
        df = pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
        frames[s] = df
    return frames

def evaluate(factor_id, fvals, closes, horizon=10, min_assets=MIN_ASSETS):
    """fvals: DataFrame (date x asset). closes: DataFrame (date x asset)."""
    fwd = closes.shift(-horizon) / closes - 1.0
    rows = []
    for dt in fvals.index:
        if dt > pd.Timestamp(FACTOR_LAST):
            continue
        f = fvals.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        n = int(mask.sum())
        if n < min_assets:
            continue
        ic, _ = spearmanr(f[mask], r[mask])
        rows.append((dt, ic, n))
    if len(rows) < 200:
        return None
    ics = pd.Series([r[1] for r in rows], index=[r[0] for r in rows])
    ic_mean = float(ics.mean()); ic_std = float(ics.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ics > 0).mean())
    # turnover: mean abs rank change over 10d rebalance
    ranks = fvals.rank(axis=1)
    r10 = ranks - ranks.shift(10)
    turn = float(r10.abs().mean(axis=1).mean())
    # coverage
    cov_asset_days = float(fvals.notna().sum().sum() / (fvals.shape[0] * fvals.shape[1]))
    cov_dates = float((fvals.notna().sum(axis=1) >= min_assets).mean())
    # decay
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        fwdh = closes.shift(-h) / closes - 1.0
        hs = []
        for dt in fvals.index:
            if dt > pd.Timestamp(FACTOR_LAST):
                continue
            f = fvals.loc[dt]; r = fwdh.loc[dt]
            m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
            if m.sum() < min_assets: continue
            ic, _ = spearmanr(f[m], r[m])
            hs.append(ic)
        decay[str(h)] = round(float(np.mean(hs)), 4) if hs else None
    return {
        "factor_id": factor_id, "ic": round(ic_mean, 4), "icir": round(icir, 4),
        "ic_hit_ratio": round(hit, 3), "n_ic_dates": len(rows),
        "ic_std": round(ic_std, 4), "turnover_10d_rank": round(turn, 3),
        "coverage_asset_days": round(cov_asset_days, 3),
        "coverage_dates_ge8": round(cov_dates, 3), "decay_ic_by_horizon": decay,
    }

def main():
    frames = load_panel()
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index()
    opens  = pd.DataFrame({s: f["open"]  for s, f in frames.items()}).sort_index()
    highs  = pd.DataFrame({s: f["high"]  for s, f in frames.items()}).sort_index()
    lows   = pd.DataFrame({s: f["low"]   for s, f in frames.items()}).sort_index()
    vols   = pd.DataFrame({s: f["volume"].astype(float) for s, f in frames.items()}).sort_index()
    print(f"panel dates: {closes.index[0].date()} .. {closes.index[-1].date()}, assets={closes.shape[1]}")

    cands = {}
    # --- volume family ---
    v20 = vols.rolling(20).mean(); v60 = vols.rolling(60).mean()
    cands["vol_ratio_20x60"] = (v20 / v60 - 1.0)
    vstd60 = vols.rolling(60).std()
    cands["vol_z_20x60"] = ((vols.rolling(20).mean() - v60) / vstd60.replace(0, np.nan))
    cands["vol_skew_60d"] = (vols.rolling(60).skew())
    # volume-return correlation 20d
    rets = closes.pct_change()
    cands["vol_ret_corr_20d"] = vols.rolling(20).corr(rets)
    # --- candle / range family ---
    hl = (highs - lows).replace(0, np.nan)
    cands["close_pos_20d"] = ((closes - lows) / hl).rolling(20).mean()
    shadow = (highs - pd.concat([opens, closes], axis=1).max(axis=1)) / hl
    cands["upper_shadow_20d"] = shadow.rolling(20).mean()
    cands["range_20d"] = (hl / closes).rolling(20).mean()
    prev_close = closes.shift(1)
    overnight = (opens - prev_close) / prev_close
    cands["overnight_20d"] = overnight.rolling(20).mean()
    cands["intraday_20d"] = ((closes - opens) / opens).rolling(20).mean()

    results = []
    for fid, fv in cands.items():
        res = evaluate(fid, fv, closes, horizon=10)
        if res:
            results.append(res)
            print(json.dumps(res))
        else:
            print(f"{fid}: INSUFFICIENT DATA")
    print("\n=== PASS GATE (|IC|>=0.007 & |ICIR|>=0.084 @h10) ===")
    for r in results:
        if abs(r["ic"]) >= 0.007 and abs(r["icir"]) >= 0.084:
            print("PASS", r["factor_id"], r["ic"], r["icir"])
        else:
            print("fail", r["factor_id"], r["ic"], r["icir"])

if __name__ == "__main__":
    main()
