"""miner_2 batch Z fix (2031-08-25) - finish batch Z screening + full validation
package for ALL IC/ICIR passers (batch Y + Z), including pairwise rho among
passers and rho vs the 3-factor live library.

Admission gates (h=10, min_valid=8): |IC| >= 0.0070 and |ICIR| >= 0.0840.
Correlation conflict threshold 0.5 per worldline pairwise signal-quality contract.
No live-account interaction.
"""
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
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()}", flush=True)

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

vol20 = rets.rolling(20).std()
vol60 = rets.rolling(60).std()
vol120 = rets.rolling(120).std()

def downside_ratio(r, win=60):
    out = {}
    for a in r.columns:
        v = r[a]
        sd = v.clip(upper=0).rolling(win).std()
        tot = v.rolling(win).std()
        out[a] = sd / (tot + 1e-12)
    return pd.DataFrame(out, index=r.index)

# ---- effective library (3 live factors) ----
live = {}
live["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / vol20
live["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
live["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

# ---- full candidate set (batch Y passers + batch Z candidates) ----
C = {}
C["downside_ratio_60d"] = downside_ratio(rets, 60)                      # +1 dir
C["beta_btc_60d"] = rolling_beta(rets, closes["BTC"].pct_change(), 60)  # +1 dir
C["drawdown_60d"] = (closes - closes.rolling(60).max()) / closes.rolling(60).max()  # -1 dir
C["hl_pos_20d"] = (closes - closes.rolling(20).min()) / (closes.rolling(20).max() - closes.rolling(20).min())  # -1 dir
vix_ret = vix.pct_change()
C["vix_beta_x_level_60d"] = -rolling_beta(rets, vix_ret, 60) * (vix / vix.shift(60)).to_frame(0).values  # +1 dir
C["mom60_skip5_voladj"] = (closes.shift(5)/closes.shift(65) - 1) / vol60   # -1 dir (observed)
C["sharpe_120d"] = (closes/closes.shift(120) - 1) / (vol120 + 1e-12)       # -1 dir (observed)
C["drawdown_120d"] = (closes - closes.rolling(120).max()) / closes.rolling(120).max()  # -1 dir (observed)
C["rev5_voladj"] = -(closes/closes.shift(5) - 1) / vol20                   # +1 dir
C["vol_ts_slope"] = (vol20 - vol60) / (vol60 + 1e-12)                      # -1 dir (observed)
C["skew_60d"] = rets.rolling(60).skew()                                    # +1 dir
C["xau_dn_beta_60d"] = rolling_beta(rets, closes["XAU"].pct_change() * (mkt_ret < 0).astype(float), 60)  # +1 dir
C["beta_usdcny_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)         # +1 dir

DIR = {"drawdown_60d": -1, "hl_pos_20d": -1, "mom60_skip5_voladj": -1,
       "sharpe_120d": -1, "drawdown_120d": -1, "vol_ts_slope": -1}

print("\n=== FULL CANDIDATE EVAL (h=10) ===", flush=True)
results = {}
for name, sig in C.items():
    d = DIR.get(name, 1)
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=d)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    dec = decay_profile(sig, closes, horizons=(1, 3, 5, 10, 20), min_valid=8, expected_sign=d)
    s.update(cov)
    s["turnover_10d_rank"] = to
    s["decay_ic_by_horizon"] = dec
    results[name] = (s, ics, sig)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:24s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"cov={s['coverage_dates_ge8']:.2f} to={to}{flag}", flush=True)

passing = {k: v for k, v in results.items() if abs(v[0]["ic"]) >= 0.0070 and abs(v[0]["icir"]) >= 0.0840}
print(f"\nFull-pass count: {len(passing)}", flush=True)

# ---- pairwise spearman rho among passers + vs live library ----
def spearman(a, b):
    both = pd.concat([a.stack().rename("a"), b.stack().rename("b")], axis=1).dropna()
    if len(both) < 30:
        return float("nan")
    return float(both["a"].corr(both["b"], method="spearman"))

names = list(passing.keys())
print("\n=== PAIRWISE SPEARMAN RHO (passers) ===", flush=True)
rho_mat = pd.DataFrame(index=names, columns=names, dtype=float)
for i in names:
    for j in names:
        if i == j:
            rho_mat.loc[i, j] = 1.0
        else:
            rho_mat.loc[i, j] = round(spearman(passing[i][2], passing[j][2]), 4)
print(rho_mat.round(3).to_string(), flush=True)

print("\n=== RHO vs LIVE LIBRARY (3 effective) ===", flush=True)
for name in names:
    row = {ln: round(spearman(passing[name][2], lsig), 4) for ln, lsig in live.items()}
    mx = max(row.values(), key=abs)
    print(f"{name:24s} {row}  max_abs={mx:.4f}", flush=True)

# ---- also max_abs_library_correlation vs FULL historical library (incl. evicted) ----
lib_full = dict(live)
lib_full["mom_10d_skip5"] = closes.shift(5) / closes.shift(15) - 1.0
lib_full["mom_120d_skip5"] = closes.shift(5) / closes.shift(125) - 1.0
lib_full["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
vix_beta = rolling_beta(rets, vix_ret, 60)
lib_full["vix_beta_cond_60x20"] = -vix_beta * (vix / vix.shift(20) - 1.0)
lib_full["vol_price_corr_20"] = rets.rolling(20).corr(mkt_ret)
lib_full["vol_ratio_20_60"] = vol20 / vol60
lib_full["rsi_14"] = 100 - 100/(1 + (rets.clip(lower=0).rolling(14).mean())/((-rets.clip(upper=0)).rolling(14).mean()+1e-9))
lib_full["us10y_cond_beta_60d"] = rolling_beta(rets, closes["US10Y"].pct_change(), 60)
lib_full["usdcny_beta_60d"] = rolling_beta(rets, usdcny.pct_change(), 60)
lib_full["eurusd_beta_60d"] = rolling_beta(rets, eurusd.pct_change(), 60)

print("\n=== MAX_ABS_LIBRARY_CORRELATION (full historical lib incl. evicted) ===", flush=True)
for name in names:
    best, key = 0.0, None
    for ln, lsig in lib_full.items():
        if ln == name:
            continue
        r = spearman(passing[name][2], lsig)
        if not np.isnan(r) and abs(r) > best:
            best, key = abs(r), ln
    print(f"{name:24s} max_abs_lib_corr={best:.4f} (vs {key})", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
