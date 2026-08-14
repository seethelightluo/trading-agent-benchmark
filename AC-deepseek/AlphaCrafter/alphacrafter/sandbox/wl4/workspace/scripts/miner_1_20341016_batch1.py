"""miner_1 2034-10-16: (1) revalidate effective library factors for drift;
(2) screen NEW candidate factor ideas (batch 1) on the 15-asset cross-asset universe.
Data visible through the previous completed trading day only (no lookahead).
Admission gates (shared): abs(IC) >= 0.0070 AND abs(ICIR) >= 0.0840 at 10d horizon.
Vectorized R2 + fast numpy rank-IC to stay within time limits.
"""
import sys, warnings, time
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
import factor_research_lib as frl
from factor_research_lib import (
    load_panels, close_panel, forward_returns,
    coverage_metrics, turnover_rank, library_signals,
)

# ---- fast rank IC (numpy, pre-ranked panels) ----
def fast_rank_ic(factor_panel, fwd, min_valid=8):
    idx = factor_panel.index.intersection(fwd.index)
    if len(idx) == 0:
        return pd.Series(dtype=float, name="ic")
    F = factor_panel.loc[idx].rank(axis=1).values.astype(float)
    R = fwd.loc[idx].rank(axis=1).values.astype(float)
    dates, ics = [], []
    for t in range(len(idx)):
        f = F[t]; r = R[t]
        m = ~(np.isnan(f) | np.isnan(r))
        if m.sum() < min_valid:
            continue
        fv = f[m]; rv = r[m]
        fs, rs = fv.std(), rv.std()
        if fs < 1e-14 or rs < 1e-14:
            continue
        ic = float(np.corrcoef(fv, rv)[0, 1])
        if not np.isnan(ic):
            dates.append(idx[t]); ics.append(ic)
    return pd.Series(ics, index=pd.DatetimeIndex(dates), name="ic")

frl.rank_ic_series = fast_rank_ic
from factor_research_lib import full_eval

t_start = time.time()
panels = load_panels(days=4000)
closes = close_panel(panels)
rets = closes.pct_change()
END = closes.index.max().strftime("%Y-%m-%d")
print(f"data through: {END} | n_dates: {len(closes)} | n_assets: {closes.shape[1]}", flush=True)

vix = panels["VIX"]["close"].astype(float) if "VIX" in panels else None
us10y = panels["US10Y"]["close"].astype(float) if "US10Y" in panels else None

hi = pd.concat({a: panels[a]["high"].astype(float) for a in closes.columns}, axis=1).sort_index()
lo = pd.concat({a: panels[a]["low"].astype(float) for a in closes.columns}, axis=1).sort_index()
vol_p = pd.concat({a: panels[a]["volume"].astype(float) for a in closes.columns}, axis=1).sort_index()

def eval_factor(name, sig, expected_sign, window=None, library=None):
    s = sig if window is None else sig.loc[window[0]:window[1]]
    c = closes if window is None else closes.loc[window[0]:window[1]]
    m, ics = full_eval(s, c, (1, 2, 3, 5, 10, 20), 8, expected_sign,
                       library=library, admission_horizon=10)
    m["admission_gate"] = {
        "ic_gate_abs": 0.0070, "icir_gate_abs": 0.0840,
        "ic_pass": abs(m["ic"]) >= 0.0070,
        "icir_pass": abs(m["icir"]) >= 0.0840,
    }
    gate = m["admission_gate"]
    ok = gate["ic_pass"] and gate["icir_pass"]
    print(f"=== {name} (dir {expected_sign:+d}) | ic={m['ic']} icir={m['icir']} "
          f"hit={m['ic_hit_ratio']} n={m['n_ic_dates']} cov={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} maxcorr={m.get('max_abs_library_correlation')} "
          f"({m.get('max_corr_factor')}) gate={'PASS' if ok else 'FAIL'}", flush=True)
    return m, ics

# ---------- library reference signals (effective + demoted) ----------
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
cn10y = panels["CN10Y"]["close"].astype(float) if "CN10Y" in panels else None
if cn10y is not None:
    cn10y_ret = cn10y.pct_change()
    beta_cn = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), cn10y_ret.rename("r")], axis=1).dropna()
        beta_cn[a] = z["a"].rolling(60).cov(z["r"]) / z["r"].rolling(60).var()
    lib_sigs["rate_beta_cn10y_60d"] = pd.DataFrame(beta_cn, index=rets.index)
eff_lib = {k: lib_sigs[k] for k in ["vol_adj_mom_accel_20x60", "dn_mkt_beta_60d", "rate_beta_cn10y_60d"]}
print("effective library reference signals:", list(eff_lib.keys()), flush=True)

print("=" * 70)
print("PART 1: REVALIDATE EFFECTIVE FACTORS (drift check)")
print("=" * 70)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, library=eff_lib)
print("--- RECENT 2Y drift (2032-10-13..END) ---", flush=True)
for nm, sg, sd in [("vol_adj_mom_accel_20x60", eff_lib["vol_adj_mom_accel_20x60"], 1),
                   ("dn_mkt_beta_60d", eff_lib["dn_mkt_beta_60d"], 1),
                   ("rate_beta_cn10y_60d", eff_lib["rate_beta_cn10y_60d"], -1)]:
    eval_factor(nm, sg, sd, ("2032-10-13", END), library=eff_lib)

print("=" * 70)
print("PART 2: NEW CANDIDATE SCREEN (batch 2034-10-16)")
print("=" * 70)

results = {}

# C1: eff_ratio_20d - Kaufman efficiency ratio: net move / gross path over 20d
net20 = (closes - closes.shift(20)).abs()
path20 = rets.abs().rolling(20).sum()
sig_er = net20 / path20.replace(0, np.nan)
m, _ = eval_factor("eff_ratio_20d", sig_er, 1, library=eff_lib); results["eff_ratio_20d"] = m

# C2: r2_trend_60d - R^2 of linear fit of log price over 60d (vectorized rolling OLS)
def rolling_r2(y: pd.DataFrame, n: int = 60):
    """Per-column rolling R^2 of log-price on time trend, vectorized via rolling sums."""
    idx = np.arange(len(y))
    k = idx.astype(float)
    ky = y.mul(k, axis=0)
    sy = y.rolling(n).sum()
    sky = ky.rolling(n).sum()
    j_center = k.reshape(-1, 1)  # end position of window
    # mean t in window ending at j: j - (n-1)/2
    mt = (idx - (n - 1) / 2.0)[:, None]
    cov = (sky / n) - mt * (sy / n)          # cov(t, y)
    var_t = (n * n - 1) / 12.0
    sy2 = (y ** 2).rolling(n).sum()
    var_y = (sy2 / n) - (sy / n) ** 2
    r2 = (cov ** 2) / (var_t * var_y)
    return r2
sig_r2 = rolling_r2(np.log(closes), 60)
m, _ = eval_factor("r2_trend_60d", sig_r2, 1, library=eff_lib); results["r2_trend_60d"] = m

# C3: bb_pos_20d - Bollinger band position (z-score of close vs its own 20d MA)
sma20 = closes.rolling(20).mean()
sig_bb = (closes - sma20) / closes.rolling(20).std().replace(0, np.nan)
m, _ = eval_factor("bb_pos_20d", sig_bb, 1, library=eff_lib); results["bb_pos_20d"] = m

# C4: downside_vol_ratio_20d - downside std / total std over 20d
neg = rets.where(rets < 0, np.nan)
down_std = neg.rolling(20).std()
tot_std = rets.rolling(20).std()
sig_dvr = down_std / tot_std.replace(0, np.nan)
m, _ = eval_factor("downside_vol_ratio_20d", sig_dvr, -1, library=eff_lib); results["downside_vol_ratio_20d"] = m

# C5: range_pos_20d - close position within 20d high-low range
rng_hi = hi.rolling(20).max()
rng_lo = lo.rolling(20).min()
sig_rp = (closes - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)
m, _ = eval_factor("range_pos_20d", sig_rp, 1, library=eff_lib); results["range_pos_20d"] = m

# C6: mom5_mom20_spread - short-horizon acceleration (mom5 - mom20)/vol20
mom5 = closes / closes.shift(5) - 1.0
sig_m5 = (mom5 - mom20) / vol20.replace(0, np.nan)
m, _ = eval_factor("mom5_mom20_spread", sig_m5, 1, library=eff_lib); results["mom5_mom20_spread"] = m

# C7: corr_us10y_20d - rolling 20d correlation of asset returns with US10Y returns
if us10y is not None:
    us10y_ret = us10y.pct_change()
    sig_cu = rets.rolling(20).corr(us10y_ret)
    m, _ = eval_factor("corr_us10y_20d", sig_cu, -1, library=eff_lib); results["corr_us10y_20d"] = m

# C8: amihud_20d - mean(|ret|/volume) over 20d (illiquidity)
sig_am = (rets.abs() / vol_p.replace(0, np.nan)).rolling(20).mean()
m, _ = eval_factor("amihud_20d", sig_am, 1, library=eff_lib); results["amihud_20d"] = m

# C9: skew_term_structure - skew(10) - skew(60)
sig_sts = rets.rolling(10).skew() - rets.rolling(60).skew()
m, _ = eval_factor("skew_term_structure", sig_sts, -1, library=eff_lib); results["skew_term_structure"] = m

# C10: xau_cond_beta_20x20 - beta(asset, XAU, 20) * mom(XAU, 20) (conditional safe-haven tilt)
xau_ret = rets["XAU"]
beta_xa = rets.rolling(20).cov(xau_ret) / xau_ret.rolling(20).var().replace(0, np.nan)
mom_xau = xau_ret.rolling(20).sum()
sig_xcb = beta_xa * mom_xau
m, _ = eval_factor("xau_cond_beta_20x20", sig_xcb, 1, library=eff_lib); results["xau_cond_beta_20x20"] = m

# C11: vix_level_cond_60 - -beta(asset, VIX, 60) * z(VIX level) (defensive tilt when VIX high)
if vix is not None:
    vix_ret = vix.pct_change()
    beta_vx = {}
    for a in rets.columns:
        z = pd.concat([rets[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
        beta_vx[a] = z["a"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
    beta_vx_df = pd.DataFrame(beta_vx, index=rets.index)
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    sig_vlc = -beta_vx_df * vix_z
    m, _ = eval_factor("vix_level_cond_60", sig_vlc, 1, library=eff_lib); results["vix_level_cond_60"] = m

# C12: max_gain_20d - max daily return over 20d (lottery preference)
sig_mg = rets.rolling(20).max()
m, _ = eval_factor("max_gain_20d", sig_mg, -1, library=eff_lib); results["max_gain_20d"] = m

# C13: mom_vol_ratio_20d - 20d momentum per unit 20d vol (Sharpe-like)
sig_mv = mom20 / vol20.replace(0, np.nan)
m, _ = eval_factor("mom_vol_ratio_20d", sig_mv, 1, library=eff_lib); results["mom_vol_ratio_20d"] = m

print("=" * 70)
print("RECENT 2Y WINDOW CHECK for candidates that PASS full-window gate")
print("=" * 70)
cand_defs = {
    "eff_ratio_20d": sig_er, "r2_trend_60d": sig_r2, "bb_pos_20d": sig_bb,
    "downside_vol_ratio_20d": sig_dvr, "range_pos_20d": sig_rp, "mom5_mom20_spread": sig_m5,
    "amihud_20d": sig_am, "skew_term_structure": sig_sts, "xau_cond_beta_20x20": sig_xcb,
    "max_gain_20d": sig_mg, "mom_vol_ratio_20d": sig_mv,
}
if us10y is not None:
    cand_defs["corr_us10y_20d"] = sig_cu
if vix is not None:
    cand_defs["vix_level_cond_60"] = sig_vlc
signed = {"downside_vol_ratio_20d": -1, "corr_us10y_20d": -1, "skew_term_structure": -1,
          "max_gain_20d": -1}
for nm, mm in results.items():
    g = mm["admission_gate"]
    if g["ic_pass"] and g["icir_pass"]:
        sd = signed.get(nm, 1)
        m2, _ = eval_factor(nm + "_RECENT2Y", cand_defs[nm], sd, ("2032-10-13", END), library=eff_lib)
        results[nm + "_recent2y"] = m2

print("=" * 70)
print("SUMMARY")
print("=" * 70)
for nm, mm in results.items():
    g = mm["admission_gate"]
    ok = "PASS" if (g["ic_pass"] and g["icir_pass"]) else "fail"
    print(f"{nm:28s} ic={mm['ic']:+.4f} icir={mm['icir']:+.4f} hit={mm['ic_hit_ratio']:.2f} "
          f"n={mm['n_ic_dates']:5d} cov8={mm['coverage_dates_ge8']:.2f} turn={mm['turnover_10d_rank']} "
          f"maxcorr={mm.get('max_abs_library_correlation')} -> {ok}", flush=True)
print("elapsed_s:", round(time.time() - t_start, 1), flush=True)
print("DONE", flush=True)
