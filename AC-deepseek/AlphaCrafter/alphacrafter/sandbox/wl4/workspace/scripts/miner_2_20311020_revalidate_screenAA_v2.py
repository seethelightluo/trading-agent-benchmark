"""miner_2 batch AA fix - rerun candidate eval with proper DataFrames + full validation for passers."""
import sys, time, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)

def align(series, idx):
    return series.reindex(idx).ffill()

vix = align(panels["VIX"]["close"].astype(float), closes.index)
dxy = align(panels["DXY"]["close"].astype(float), closes.index)
usdjpy = align(panels["USDJPY"]["close"].astype(float), closes.index)
eurusd = align(panels["EURUSD"]["close"].astype(float), closes.index)
usdcny = align(panels["USDCNY"]["close"].astype(float), closes.index)

H = 10
fwd = forward_returns(closes, H)

def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

def report(name, sig, expected_sign=1):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    recent = {}
    for w in (63, 126, 252, 504):
        sub = ics.iloc[-w:]
        if len(sub) > 2:
            mm, ss = sub.mean(), sub.std(ddof=1)
            recent[w] = (mm, mm/ss if ss and ss > 0 else np.nan)
        else:
            recent[w] = (np.nan, np.nan)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"| r63=({recent[63][0]:+.3f},{recent[63][1]:+.2f}) r126=({recent[126][0]:+.3f},{recent[126][1]:+.2f}) "
          f"r252=({recent[252][0]:+.3f},{recent[252][1]:+.2f}) r504=({recent[504][0]:+.3f},{recent[504][1]:+.2f}) "
          f"cov={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics

vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vol120 = rets.rolling(120).std()

# remaining candidates (DataFrame-form)
AA = {}
AA["vix_level_z"] = pd.DataFrame({a: ((vix - vix.rolling(60).mean()) / (vix.rolling(60).std() + 1e-12)).values
                                  for a in closes.columns}, index=closes.index)
AA["dxy_mom20"] = pd.DataFrame({a: (dxy/dxy.shift(20) - 1).values for a in closes.columns}, index=closes.index)
# extra candidates to widen search
AA["vix_beta_x_level_60d"] = -rolling_beta(rets, vix.pct_change(), 60) * (vix / vix.shift(60)).to_frame(0).values
AA["hl_pos_20d"] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min() + 1e-12)
AA["downside_ratio_60d"] = pd.DataFrame({a: rets[a].clip(upper=0).rolling(60).std() / (rets[a].rolling(60).std() + 1e-12)
                                         for a in rets.columns}, index=rets.index)
AA["skew_60d"] = rets.rolling(60).skew()
AA["kurt_60d"] = rets.rolling(60).kurt()
AA["beta_btc_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)
AA["beta_eth_60d"] = rolling_beta(rets, closes["ETH"].pct_change(), 60)
AA["corr_asset_mkt_20"] = rets.rolling(20).corr(mkt_ret)
AA["corr_asset_mkt_60"] = rets.rolling(60).corr(mkt_ret)
AA["max_dd_60d"] = (closes - closes.rolling(60).max()) / closes.rolling(60).max()
AA["mom60_skip5_voladj"] = (closes.shift(5)/closes.shift(65) - 1) / vol60

results = {}
for name, sig in AA.items():
    s, ics = report(name, sig, expected_sign=1)
    results[name] = (s, ics, sig)

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"\nFull-pass count (batch AA v2): {len(passing)}", flush=True)

existing = {}
existing["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
existing["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
existing["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

def spearman_rho_vs_live(cand, library):
    best, best_key = 0.0, None
    for name, lsig in library.items():
        both = pd.concat([cand.stack().rename("c"), lsig.stack().rename("l")], axis=1).dropna()
        if len(both) < 30:
            continue
        rr = float(both["c"].corr(both["l"], method="spearman"))
        if abs(rr) > best:
            best, best_key = abs(rr), name
    return round(best, 4), best_key

for name, (s, ics, sig) in passing.items():
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=1)
    rho, key = spearman_rho_vs_live(sig, existing)
    print(f"{name:26s} decay={dec} rho_vs_live={rho:.4f} ({key})", flush=True)

out = {k: {kk: vv for kk, vv in v[0].items()} for k, v in results.items()}
out["_meta"] = {"asof": str(closes.index.max().date()), "n_assets": closes.shape[1],
                "gates": {"abs_ic": 0.0070, "abs_icir": 0.0840, "min_valid": 8, "h": H}}
with open("scripts/_miner2_20311020_batchAA_v2_results.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(f"\nsaved scripts/_miner2_20311020_batchAA_v2_results.json | done {time.time()-t0:.1f}s", flush=True)
