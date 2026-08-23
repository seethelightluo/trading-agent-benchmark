"""miner_2 screen batch (2035-06-20). Fresh candidates for the 15-instrument cross-asset
universe under CURRENT regime: VIX has fallen from >60 (risk-off) to ~22 (risk-on).
Admission gate (benchmark-wide): |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Only data visible through 2035-06-19. Reports n dates / instruments actually used.
"""
import sys, time
sys.path.insert(0, 'scripts')
from factor_validation_lib import load_panel, load_macro, ic_analysis, print_report
import pandas as pd, numpy as np
from scipy.stats import rankdata

VIS = "2035-06-19"
px = load_panel(max_date=VIS)
print("panel:", px.shape, "n_dates", px.shape[0], "n_assets", px.shape[1], flush=True)
ret = px.pct_change()
vix = load_macro("VIX", max_date=VIS)
print("VIX regime recent:", vix.tail(5).round(1).tolist(), flush=True)

def evalc(f, label):
    res = ic_analysis(f, px, horizon=10, label=label)
    print_report(res)
    ic = res["ic"]; icir = res["icir"]
    gate = bool(ic is not None and icir is not None and abs(ic) >= 0.0070 and abs(icir) >= 0.0840)
    print("  => GATE PASS\n" if gate else "  => GATE FAIL\n")
    return res, gate

def build_library():
    lib = {}
    vix_ret = vix.pct_change()
    spx_ret = px["SPX"].pct_change()
    lib["beta_vix_60d_neg"] = -ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
    lib["mom_10d_skip5"] = (px / px.shift(10) - 1).rank(axis=1, pct=True)
    lib["mom_120d_skip5"] = (px / px.shift(120) - 1).rank(axis=1, pct=True)
    lib["vol_beta_spx_60d"] = ret.rolling(60).cov(spx_ret) / spx_ret.rolling(60).var()
    lib["sign_ewma_60d"] = np.sign(ret).ewm(span=60).mean()
    lib["down_vol_ratio_20x120"] = ret.clip(upper=0).rolling(20).std() / ret.rolling(120).std()
    lib["skew_20d_neg"] = -ret.rolling(20).skew()
    lib["vol_of_vol20x60"] = ret.rolling(20).std() / ret.rolling(60).std()
    return lib

lib = build_library()

def lib_corr(f):
    f_rank = f.rank(axis=1, pct=True)
    best = 0.0
    for fid, sig in lib.items():
        s = sig.reindex(f_rank.index).rank(axis=1, pct=True)
        row_ics = []
        for d in f_rank.index.intersection(s.index):
            a, b = f_rank.loc[d], s.loc[d]
            m = a.notna() & b.notna()
            if m.sum() >= 8:
                v1, v2 = rankdata(a[m].values), rankdata(b[m].values)
                if np.std(v1) == 0 or np.std(v2) == 0:
                    continue
                row_ics.append(np.corrcoef(v1, v2)[0, 1])
        if len(row_ics):
            best = max(best, abs(float(np.mean(row_ics))))
    return round(best, 3)

cands = {}
mom5 = px / px.shift(5) - 1
mom10 = px / px.shift(10) - 1
mom20 = px / px.shift(20) - 1

# A. 20d momentum (fresh horizon vs library mom10/mom120)
cands["mom_20d_skip5"] = mom20.rank(axis=1, pct=True)

# B. VIX-fall conditioned short-term mean reversion (low VIX regime favors catch-up)
vix_lvl = vix.reindex(px.index, method='ffill')
vix_ratio = vix_lvl / vix_lvl.rolling(60).mean()
cands["vix_fall_catchup5"] = (-mom5.rank(axis=1, pct=True)) * (1.0 - vix_ratio.clip(0, 2))

# C. Vol-adjusted 5d momentum (risk-on appetite)
cands["vol_adj_mom5"] = ret.rolling(5).sum() / ret.rolling(10).std()

# D. Up-down vol imbalance (upside vs downside vol spread, fresh angle)
up = ret.clip(lower=0).rolling(20).std()
dn = ret.clip(upper=0).rolling(20).std()
tot = ret.rolling(60).std()
cands["updown_vol_imbalance20"] = (up - dn) / tot

# E. Trend consistency over 60d demeaned
pos60 = (ret > 0).rolling(60).mean()
cands["consistency60_cs"] = pos60 - pos60.mean(axis=1)

# F. 10d spread-to-vol (information-adjusted momentum ratio)
spread = px.rolling(10).apply(lambda x: x.max()-x.min(), raw=True)
cands["range_vol_ratio10"] = spread / (ret.rolling(10).std() * px)

passed = []
for name, f in cands.items():
    try:
        res, gate = evalc(f, name)
        if gate:
            lc = lib_corr(f)
            res["max_abs_library_correlation"] = lc
            passed.append((name, res, lc))
            print(f"  [lib_corr for {name}] = {lc}", flush=True)
    except Exception as e:
        print(f"[{name}] ERROR {e}", flush=True)

print("\n===== FINAL =====", flush=True)
for name, res, lc in passed:
    print(f"{name}: IC={res['ic']} ICIR={res['icir']} hit={res['ic_hit_ratio']} "
          f"n_dates={res['n_ic_dates']} cov_dates={res['coverage_dates_ge8']} "
          f"turn={res['turnover_10d_rank']} lib_corr={lc} decay={res['decay_ic_by_horizon']}", flush=True)
if not passed:
    print("No candidate passed the gate.", flush=True)