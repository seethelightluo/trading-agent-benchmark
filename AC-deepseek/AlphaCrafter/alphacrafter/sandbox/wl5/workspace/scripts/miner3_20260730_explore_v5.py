"""miner_3 2026-07-30: explore v5 - novel factor family screening.

Candidate ideas (all cross-asset, 15-instrument universe, h=10 horizon):
  F1 range_pos_20       : 20d avg of daily close location within day's [low,high] range
  F2 trend_r2_60        : R^2 of log-price on time trend (trend consistency/quality)
  F3 downside_vol_ratio : 60d downside semi-dev / total vol (loss asymmetry)
  F4 autocorr_20        : lag-1 autocorrelation of daily returns (trend persistence)
  F5 vol_term_10x60     : 10d realized vol / 60d realized vol (vol regime)
  F6 usdjpy_beta_60     : rolling 60d beta to USDJPY returns (macro obs-only)
  F7 eurusd_beta_60     : rolling 60d beta to EURUSD returns (macro obs-only)
  F8 vol_price_corr_60  : corr(daily return, volume) over 60d (volume-confirmed trend)
  F9 hl_range_ratio_20  : 20d avg (high-low)/close scaled by 20d vol (intraday info)
  F10 drawdown_120      : close / rolling_max(close,120) - 1 (distance to peak)
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
ret = close.pct_change()
fr = forward_returns(close, H)

# volume + OHLC panel
frames = {s: pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
          for s in WATCH}
vol = pd.DataFrame({s: f.set_index("date")["volume"].astype(float)
                    for s, f in frames.items()}).sort_index()
hi = pd.DataFrame({s: f.set_index("date")["high"].astype(float)
                   for s, f in frames.items()}).sort_index()
lo = pd.DataFrame({s: f.set_index("date")["low"].astype(float)
                   for s, f in frames.items()}).sort_index()
hi = hi[hi.index <= pd.Timestamp(VIS)]
lo = lo[lo.index <= pd.Timestamp(VIS)]
vol = vol[vol.index <= pd.Timestamp(VIS)]

macro = pd.DataFrame({
    "USDJPY": pd.read_csv("../persistent/index_data/USDJPY.csv", parse_dates=["date"]).set_index("date")["close"],
    "EURUSD": pd.read_csv("../persistent/index_data/EURUSD.csv", parse_dates=["date"]).set_index("date")["close"],
}).sort_index()
macro = macro[macro.index <= pd.Timestamp(VIS)]
macro_ret = macro.pct_change()


def rolling_beta(asset_ret, mkt_ret, win, minp=40):
    out = {}
    for a in asset_ret.columns:
        pair = pd.concat([asset_ret[a].rename("a"), mkt_ret.rename("m")], axis=1).dropna()
        b = pair["a"].rolling(win, min_periods=minp).cov(pair["m"]) / pair["m"].rolling(win, min_periods=minp).var()
        out[a] = b
    return pd.DataFrame(out).reindex(asset_ret.index)


signals = {}

# F1 range position: avg of (close-low)/(high-low) over 20d
day_pos = (close - lo) / (hi - lo).replace(0, np.nan)
signals["range_pos_20"] = day_pos.rolling(20).mean()

# F2 trend R^2 over 60d
def trend_r2(px, win=60, minp=40):
    out = {}
    for a in px.columns:
        s = px[a]
        def r2(x):
            if len(x) < minp:
                return np.nan
            y = np.log(x.values)
            t = np.arange(len(x))
            if np.std(y) == 0:
                return np.nan
            return float(np.corrcoef(t, y)[0, 1] ** 2)
        out[a] = s.rolling(win, min_periods=minp).apply(r2, raw=True)
    return pd.DataFrame(out)
signals["trend_r2_60"] = trend_r2(close, 60, 40)

# F3 downside vol ratio: semi-dev(neg rets) / total vol, 60d
neg = ret.where(ret < 0, 0.0)
down_sd = np.sqrt((neg ** 2).rolling(60).mean())
tot_sd = ret.rolling(60).std()
signals["downside_vol_ratio_60"] = down_sd / tot_sd

# F4 lag-1 autocorrelation of returns over 20d
def autocorr20(x):
    if x.notna().sum() < 15:
        return np.nan
    x = x.dropna()
    if x.std() == 0:
        return np.nan
    return float(x.autocorr(lag=1))
signals["autocorr_20"] = ret.rolling(20, min_periods=15).apply(autocorr20, raw=False)

# F5 vol term structure: 10d vol / 60d vol
signals["vol_term_10x60"] = ret.rolling(10).std() / ret.rolling(60).std()

# F6/F7 macro beta
signals["usdjpy_beta_60"] = rolling_beta(ret, macro_ret["USDJPY"], 60)
signals["eurusd_beta_60"] = rolling_beta(ret, macro_ret["EURUSD"], 60)

# F8 volume-price correlation over 60d
logv = np.log(vol.replace(0, np.nan))
signals["vol_price_corr_60"] = ret.rolling(60).corr(logv)

# F9 high-low range ratio
hl = ((hi - lo) / close).rolling(20).mean()
signals["hl_range_ratio_20"] = hl / ret.rolling(20).std()

# F10 drawdown depth vs 120d high
signals["drawdown_120"] = close / close.rolling(120).max() - 1.0

print(f"universe: {len(WATCH)} instruments, data window {close.index[0].date()}..{close.index[-1].date()}, horizon h={H}")
print("=" * 100)
results = {}
for fid, sig in signals.items():
    sig = sig.reindex(close.index)
    ics = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ics, sig, fr, close, h=H)
    if m is None:
        print(f"{fid:24s} INSUFFICIENT IC dates ({len(ics)})")
        continue
    reg = regime_split(ics)
    results[fid] = (sig, ics, m, reg)
    gate_ic = abs(m["ic"]) >= 0.007
    gate_icir = abs(m["icir"] or 0) >= 0.084
    flag = "*** PASS ***" if (gate_ic and gate_icir) else ""
    print(f"{fid:24s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} covA={m['coverage_asset_days']:.3f} covD8={m['coverage_dates_ge8']:.3f} "
          f"turn={m['turnover_10d_rank']:.3f} | regimes: 20-22 {reg.get('2020-2022',{}).get('ic')} "
          f"23-24 {reg.get('2023-2024',{}).get('ic')} 25-26 {reg.get('2025-2026',{}).get('ic')} {flag}")
    print("   decay:", json.dumps(m["decay_ic_by_horizon"]))

with open("scripts/miner3_20260730_explore_v5_results.json", "w") as f:
    json.dump({k: {"ic": v[2]["ic"], "icir": v[2]["icir"], "hit": v[2]["ic_hit_ratio"],
                   "n": v[2]["n_ic_dates"], "regime": v[3]} for k, v in results.items()}, f, indent=1, default=str)
print("saved results json")
