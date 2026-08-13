"""miner_2 2034-05-15: (1) revalidate 3 effective library factors for drift;
(2) screen NEW candidate factor ideas on the 15-asset cross-asset universe.
Data visible through 2034-05-12 (previous completed trading day)."""
import sys, warnings, json
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (
    load_panels, close_panel, forward_returns,
    rank_ic_series, summarize_ic, coverage_metrics, turnover_rank, decay_profile,
    full_eval, library_signals, max_library_corr,
)

panels = load_panels(days=6000)
closes = close_panel(panels)
rets = closes.pct_change()
END = closes.index.max().strftime("%Y-%m-%d")
print("data through:", END, "| n_dates:", len(closes), "| n_assets:", closes.shape[1])

vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
dxy = panels["DXY"]["close"].astype(float) if "DXY" in panels else None
us10y = panels["US10Y"]["close"].astype(float) if "US10Y" in panels else None
cn10y = panels["CN10Y"]["close"].astype(float) if "CN10Y" in panels else None

def eval_factor(name, sig, expected_sign, window=None, library=None):
    s = sig if window is None else sig.loc[window[0]:window[1]]
    c = closes if window is None else closes.loc[window[0]:window[1]]
    m, ics = full_eval(s, c, (1,2,3,5,10,20), 8, expected_sign,
                       library=library, admission_horizon=10)
    m["admission_gate"] = {
        "ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
        "ic_pass": abs(m["ic"]) >= 0.0070,
        "icir_pass": abs(m["icir"]) >= 0.0840,
        "corr_gate_abs": 0.50,
    }
    print(f"=== {name} (expected dir {expected_sign:+d}) | ic={m['ic']} icir={m['icir']} "
          f"hit={m['ic_hit_ratio']} n={m['n_ic_dates']} cov={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')} "
          f"({m.get('max_corr_factor')}) gate={m['admission_gate']['ic_pass'] and m['admission_gate']['icir_pass']}")
    return m, ics

# ---------- library reference signals ----------
lib_sigs = library_signals(panels, closes, rets, vix)
mom20 = closes / closes.shift(20) - 1.0
mom60 = closes / closes.shift(60) - 1.0
vol20 = rets.rolling(20).std()
lib_sigs["vol_adj_mom_accel_20x60"] = (mom20 - mom60) / vol20
mkt_ret = rets.mean(axis=1)
down = mkt_ret.where(mkt_ret < 0)
beta_down = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), down.rename("m")], axis=1).dropna()
    beta_down[a] = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
lib_sigs["dn_mkt_beta_60d"] = pd.DataFrame(beta_down, index=rets.index)
cn10y_ret = cn10y.pct_change()
beta_cn = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("r")], axis=1).dropna()
    beta_cn[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
lib_sigs["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)
print("library reference signals:", list(lib_sigs.keys()))

print("=" * 70)
print("PART 1: REVALIDATE 3 EFFECTIVE FACTORS (drift check)")
print("=" * 70)
eval_factor("vol_adj_mom_accel_20x60", lib_sigs["vol_adj_mom_accel_20x60"], 1, library=lib_sigs)
eval_factor("dn_mkt_beta_60d", lib_sigs["dn_mkt_beta_60d"], 1, library=lib_sigs)
eval_factor("rate_beta_cn10y_60d", lib_sigs["rate_beta_cn10y_60d"], -1, library=lib_sigs)
print("--- RECENT 1Y (2033-05-13..END) drift ---")
eval_factor("vol_adj_mom_accel_20x60", lib_sigs["vol_adj_mom_accel_20x60"], 1, ("2033-05-13", END))
eval_factor("dn_mkt_beta_60d", lib_sigs["dn_mkt_beta_60d"], 1, ("2033-05-13", END))
eval_factor("rate_beta_cn10y_60d", lib_sigs["rate_beta_cn10y_60d"], -1, ("2033-05-13", END))

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN (batch 2)")
print("=" * 70)

# N1: trend_strength_60d - trend efficiency |mom60|/(sqrt(60)*vol60)
sig_tse = mom60.abs() / (np.sqrt(60) * rets.rolling(60).std())
eval_factor("trend_strength_60d", sig_tse, 1, library=lib_sigs)

# N2: sharpe_60d - risk-adjusted mean return over 60d
sig_sharpe = rets.rolling(60).mean() / rets.rolling(60).std()
eval_factor("sharpe_60d", sig_sharpe, 1, library=lib_sigs)

# N3: win_rate_20d - fraction of positive days in last 20 (consistency)
sig_wr = (rets > 0).rolling(20).mean()
eval_factor("win_rate_20d", sig_wr, 1, library=lib_sigs)

# N4: ema_cross_12_26_voladj - MACD-style (EMA12-EMA26) normalized by vol20
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()
ema12 = closes.apply(lambda x: ema(x, 12))
ema26 = closes.apply(lambda x: ema(x, 26))
sig_ema = (ema12 - ema26) / ema26 / vol20
eval_factor("ema_cross_12_26_voladj", sig_ema, 1, library=lib_sigs)

# N5: risk_on_beta_60d - beta to crypto composite (BTC+ETH equal weight)
crypto_ret = rets[["BTC", "ETH"]].mean(axis=1)
beta_cr = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), crypto_ret.rename("m")], axis=1).dropna()
    beta_cr[a] = z["a"].rolling(60).cov(z["m"]) / z["m"].rolling(60).var()
sig_cr = pd.DataFrame(beta_cr, index=rets.index)
eval_factor("risk_on_beta_60d", sig_cr, 1, library=lib_sigs)

# N6: us10y_beta_60d - unconditional beta to US10Y returns
us10y_ret = us10y.pct_change()
beta_10y = {}
for a in rets.columns:
    z = pd.concat([rets[a].rename("a"), us10y_ret.rename("r")], axis=1).dropna()
    beta_10y[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
sig_10y = pd.DataFrame(beta_10y, index=rets.index)
eval_factor("us10y_beta_60d", sig_10y, -1, library=lib_sigs)

# N7: range_expansion_20d - (high20-low20)/close (expanding range)
hi20 = closes.rolling(20).max()
lo20 = closes.rolling(20).min()
sig_range = (hi20 - lo20) / closes
eval_factor("range_expansion_20d", sig_range, -1, library=lib_sigs)

# N8: autocorr_1d_30w - 1-day lag autocorrelation of daily returns (short-term reversal)
sig_ac1 = rets.apply(lambda x: x.rolling(30).apply(lambda y: pd.Series(y).autocorr(1), raw=False))
eval_factor("autocorr_1d_30w", sig_ac1, -1, library=lib_sigs)

# N9: dist_from_high_60d - close/rolling_max(close,60)-1 (trend position)
sig_dfh = closes / closes.rolling(60).max() - 1.0
eval_factor("dist_from_high_60d", sig_dfh, 1, library=lib_sigs)

# N10: updown_vol_ratio_60d - std(pos-day rets)/std(neg-day rets)
up_r = rets.where(rets > 0, np.nan)
dn_r = rets.where(rets < 0, np.nan)
sig_ud = up_r.rolling(60).std() / dn_r.rolling(60).std()
eval_factor("updown_vol_ratio_60d", sig_ud, 1, library=lib_sigs)

# N11: mom20_z - (close/sma20-1)/std20, z-score style (sign test both)
sma20 = closes.rolling(20).mean()
sig_z = (closes / sma20 - 1.0) / vol20
eval_factor("mom20_zscore", sig_z, 1, library=lib_sigs)

# N12: mom60_voladj - 60d momentum scaled by 60d vol (slower risk-adj momentum)
sig_mv60 = mom60 / rets.rolling(60).std()
eval_factor("mom60_voladj60", sig_mv60, 1, library=lib_sigs)

print("DONE")
