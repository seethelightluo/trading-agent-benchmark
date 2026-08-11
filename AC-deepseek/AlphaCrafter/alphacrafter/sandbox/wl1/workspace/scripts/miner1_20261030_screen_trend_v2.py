"""miner_1 cycle (2026-10-30): screen trend/momentum guard v2 + new factor families.

Motivation: memory feedback (2026-09-10) - reversal drag persisted 3 consecutive
blocks; ensemble needs a trend/momentum guard. Previous screens (miner1 2026-10-08,
miner2 2026-10-30) found most short/medium momentum variants FAIL the daily-IC gate.
This cycle tests fresh interpretable families:
  - 52-week high proximity        (hi52_252)
  - trend strength R^2 (60d)      (trend_r2_60)
  - MA50/MA200 golden-cross       (ma50_200)
  - vol-adjusted 60d momentum     (vadj_mom60_20)
  - relative momentum vs basket   (rel_mom20_basket, rel_mom60_basket)
  - short/long vol ratio          (vol_ratio_5_60)
  - downside-risk concentration   (down_ratio_20)
  - VIX-regime conditional momentum (vixz_mom20)
References: rev_1d / nclv_1d / mom120_skip5 (library signals) for regime context.

Gate: |daily rank IC| >= 0.0070 and |daily ICIR| >= 0.0840 on the 15-name panel;
a date needs >= 8 valid names. Data visible through 2026-10-29.
"""
import numpy as np
import pandas as pd

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
END = pd.Timestamp("2026-10-29")
START_FULL = pd.Timestamp("2021-01-01")
START_REC = pd.Timestamp("2026-01-01")
IC_MIN, ICIR_MIN = 0.0070, 0.0840
MIN_NAMES = 8

# ---- build fresh panel from CSVs (source of truth), no data after END ----
panel = {}
for s in SYMBOLS:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    panel[s] = df
macro = {}
for s in ["VIX", "DXY"]:
    df = pd.read_csv(f"../persistent/index_data/{s}.csv", parse_dates=["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    macro[s] = df

dates = sorted(set().union(*[set(df.index) for df in panel.values()]))
dates = pd.DatetimeIndex(dates)
close = pd.DataFrame({s: panel[s]["close"] for s in SYMBOLS}).reindex(dates)
open_ = pd.DataFrame({s: panel[s]["open"] for s in SYMBOLS}).reindex(dates)
high = pd.DataFrame({s: panel[s]["high"] for s in SYMBOLS}).reindex(dates)
low = pd.DataFrame({s: panel[s]["low"] for s in SYMBOLS}).reindex(dates)
vix = macro["VIX"]["close"].reindex(dates)
dxy = macro["DXY"]["close"].reindex(dates)
idx = close.index
print("panel dates:", idx.min().date(), "->", idx.max().date(), "rows:", len(idx))


def fast_ic(factor_df, fwd, min_names=MIN_NAMES):
    F = factor_df.values.astype(float); R = fwd.values.astype(float)
    n = np.isfinite(F) & np.isfinite(R)
    ok = n.sum(axis=1) >= min_names
    if not ok.any():
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    Fm = np.where(n, F, 0.0); Rm = np.where(n, R, 0.0)
    cnt = n.sum(axis=1)[ok]
    sx = Fm[ok].sum(axis=1); sy = Rm[ok].sum(axis=1)
    sxx = (Fm[ok] ** 2).sum(axis=1); syy = (Rm[ok] ** 2).sum(axis=1)
    sxy = (Fm[ok] * Rm[ok]).sum(axis=1)
    with np.errstate(all="ignore"):
        num = cnt * sxy - sx * sy
        den = np.sqrt((cnt * sxx - sx * sx) * (cnt * syy - sy * sy))
        ic = num / den
    ic = ic[np.isfinite(ic)]
    if len(ic) == 0:
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    return {"n_dates": int(len(ic)), "n_obs": int(cnt.sum()),
            "ic": float(ic.mean()),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
            "hit": float((ic > 0).mean())}


def turnover10(factor_df, rebal=10):
    F = factor_df.values.astype(float)
    rk = pd.DataFrame(np.argsort(np.argsort(F, axis=1), axis=1), index=factor_df.index,
                      columns=factor_df.columns).astype(float)
    rk = rk.replace(len(factor_df.columns), np.nan)
    out = {}
    for h in (5, 10):
        d = rk.diff(h).abs().mean().mean()
        out[h] = float(d) if np.isfinite(d) else np.nan
    return out


# ---- factor definitions (all causal; only past data) ----
lret = np.log(close).diff()
basket20 = np.log(close.shift(5) / close.shift(25)).mean(axis=1)
basket60 = np.log(close.shift(20) / close.shift(80)).mean(axis=1)

F = {}
# reference / library-like signals
F["rev_1d"] = -(np.log(close) - np.log(close.shift(1)))                       # library ref
F["nclv_1d"] = -(close - low) / (high - low)                                  # library ref
F["mom120_skip5"] = np.log(close.shift(5) / close.shift(125))                 # library ref
# trend / momentum guard family
F["mom20_skip5"] = np.log(close.shift(5) / close.shift(25))
F["mom60_skip20"] = np.log(close.shift(20) / close.shift(80))
F["hi52_252"] = close / close.rolling(252).max() - 1.0
F["ma50_200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1.0
F["vadj_mom60_20"] = np.log(close.shift(20) / close.shift(80)) / lret.rolling(60).std()
F["rel_mom20_basket"] = np.log(close.shift(5) / close.shift(25)).sub(basket20, axis=0)
F["rel_mom60_basket"] = np.log(close.shift(20) / close.shift(80)).sub(basket60, axis=0)

# trend strength: R^2 of log-price linear fit over trailing 60d
def trend_r2(df, win=60):
    out = pd.Series(np.nan, index=df.index)
    lp = np.log(df)
    x = np.arange(win, dtype=float)
    xm = x.mean()
    xd = ((x - xm) ** 2).sum()
    for i in range(win - 1, len(df)):
        y = lp.iloc[i - win + 1: i + 1].values
        if not np.isfinite(y).all():
            continue
        ym = y.mean()
        b = ((x - xm) @ (y - ym)) / xd
        ss_tot = ((y - ym) ** 2).sum()
        ss_res = ((y - (ym + b * (x - xm))) ** 2).sum()
        out.iloc[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return out

F["trend_r2_60"] = trend_r2(close, 60)
F["vol_ratio_5_60"] = lret.rolling(5).std() / lret.rolling(60).std() - 1.0
F["down_ratio_20"] = lret.clip(upper=0).rolling(20).std() / lret.rolling(20).std() - 1.0
# VIX-regime conditional momentum: momentum signed by VIX z-score (low vol -> momentum)
vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
vix_regime = np.where(vix_z < 0, 1.0, -1.0)
F["vixz_mom20"] = np.log(close.shift(5) / close.shift(25)).mul(pd.Series(vix_regime, index=idx), axis=0)

fwd1 = np.log(close.shift(-1)) - np.log(close)
fwd5 = np.log(close.shift(-5)) - np.log(close)
fwd10 = np.log(close.shift(-10)) - np.log(close)

rows = []
for name, fac in F.items():
    fac = fac.replace([np.inf, -np.inf], np.nan)
    for label, lo in (("full", START_FULL), ("recent", START_REC)):
        m = (idx >= lo) & (idx <= END)
        fsub = fac[m]
        cov = float(fsub.notna().mean().mean())
        r1 = fast_ic(fsub, fwd1[m]); r5 = fast_ic(fsub, fwd5[m]); r10 = fast_ic(fsub, fwd10[m])
        to = turnover10(fsub)
        rows.append({"factor": name, "sample": label, "cov": cov,
                     "ic1": r1["ic"], "icir1": r1["icir"], "hit1": r1["hit"], "n1": r1["n_dates"],
                     "ic5": r5["ic"], "icir5": r5["icir"], "n5": r5["n_dates"],
                     "ic10": r10["ic"], "icir10": r10["icir"], "n10": r10["n_dates"],
                     "turn5": to[5], "turn10": to[10]})

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(res.to_string(index=False))

full = res[res.sample == "full"].copy()
recent = res[res.sample == "recent"].copy()
def passes(r):
    return ((r.ic1.abs() >= IC_MIN) & (r.icir1.abs() >= ICIR_MIN)) | \
           ((r.ic5.abs() >= IC_MIN) & (r.icir5.abs() >= ICIR_MIN)) | \
           ((r.ic10.abs() >= IC_MIN) & (r.icir10.abs() >= ICIR_MIN))
print("\nPASS(full):", full[passes(full)].factor.tolist())
print("PASS(recent):", recent[passes(recent)].factor.tolist())
print("\nGate:", IC_MIN, ICIR_MIN, "| min names/date:", MIN_NAMES, "| universe:", len(SYMBOLS))
