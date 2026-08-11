"""miner_1 cycle (2026-10-08): screen TREND/MOMENTUM guard families.

Motivation: memory feedback (2026-09-10) - reversal drag persisted 3 consecutive
blocks; screener should trim reversal-family share and add a trend/momentum guard.
Candidates: momentum skip variants, MA-slope/trend-strength, risk-adjusted trend,
MACD, vol-compression, cross-asset equity-beta trend.

Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 (daily paper rank IC on 15-name panel).
We report IC at horizons 1/5/10 for full sample 2021-01-01..2026-10-07 and the
recent regime 2026-01-01..2026-10-07 (visible data ends 2026-10-07).
"""
import numpy as np
import pandas as pd

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
END = pd.Timestamp("2026-10-07")
START_FULL = pd.Timestamp("2021-01-01")
START_REC = pd.Timestamp("2026-01-01")
IC_MIN, ICIR_MIN = 0.0070, 0.0840
MIN_NAMES = 8

# ---- build fresh panel from CSVs (source of truth), no future data ----
panel = {}
for s in SYMBOLS:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    panel[s] = df
dates = sorted(set().union(*[set(df.index) for df in panel.values()]))
dates = pd.DatetimeIndex(dates)
close = pd.DataFrame({s: panel[s]["close"] for s in SYMBOLS}).reindex(dates).ffill()
open_ = pd.DataFrame({s: panel[s]["open"] for s in SYMBOLS}).reindex(dates).ffill()
high = pd.DataFrame({s: panel[s]["high"] for s in SYMBOLS}).reindex(dates).ffill()
low = pd.DataFrame({s: panel[s]["low"] for s in SYMBOLS}).reindex(dates).ffill()
vol = pd.DataFrame({s: panel[s]["volume"] for s in SYMBOLS}).reindex(dates).ffill()
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
    for h in (1, 5, 10):
        d = rk.diff(h).abs().mean().mean()
        out[h] = float(d) if np.isfinite(d) else np.nan
    return out


# ---- factor definitions (all causal; only past data) ----
F = {}
lret = np.log(close).diff()
F["mom20_skip5"] = np.log(close.shift(5) / close.shift(25))
F["mom30_skip10"] = np.log(close.shift(10) / close.shift(40))
F["mom60_skip20"] = np.log(close.shift(20) / close.shift(80))
F["ma20_ma60"] = close.rolling(20).mean() / close.rolling(60).mean() - 1.0
F["trend60_clv"] = (close - low.rolling(60).min()) / (high.rolling(60).max() - low.rolling(60).min()) - 0.5
F["vadj_mom20"] = np.log(close.shift(5) / close.shift(25)) / lret.rolling(20).std()
F["macd12_26"] = (close.ewm(span=12, adjust=False).mean() / close.ewm(span=26, adjust=False).mean() - 1.0)
F["vol_comp_20_60"] = lret.rolling(20).std() / lret.rolling(60).std() - 1.0
F["ma20_slope20"] = close.rolling(20).mean() / close.rolling(20).mean().shift(20) - 1.0
F["hilo_break60"] = (close - low.rolling(60).min()) / (high.rolling(60).max() - low.rolling(60).min())

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
        r1, r5, r10 = (fast_ic(fsub, fwd1[m]), fast_ic(fsub, fwd5[m]), fast_ic(fsub, fwd10[m]))
        to = turnover10(fsub)
        rows.append({"factor": name, "sample": label, "cov": cov,
                     "ic1": r1["ic"], "icir1": r1["icir"], "hit1": r1["hit"], "n1": r1["n_dates"],
                     "ic5": r5["ic"], "icir5": r5["icir"],
                     "ic10": r10["ic"], "icir10": r10["icir"],
                     "turn1": to[1], "turn10": to[10]})

res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(res.to_string(index=False))
print("\nPASS(full):", res[(res.sample == "full") &
      ((res.ic1.abs() >= IC_MIN) & (res.icir1.abs() >= ICIR_MIN)) |
      ((res.ic5.abs() >= IC_MIN) & (res.icir5.abs() >= ICIR_MIN)) |
      ((res.ic10.abs() >= IC_MIN) & (res.icir10.abs() >= ICIR_MIN))].factor.tolist())
