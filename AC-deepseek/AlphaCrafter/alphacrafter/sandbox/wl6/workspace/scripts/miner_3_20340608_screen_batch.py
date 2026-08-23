"""miner_3 screen batch (2034-06-08), VIS=2034-06-07.
Fresh candidate factors for the 15-instrument cross-asset universe.
Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
Avoids re-testing persisted ideas (mom, vol, beta-defensive, skew, xau_copper...).
"""
import sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE, ic_analysis, library_corr, load_panel
import pandas as pd, numpy as np, math, json

VIS = "2034-06-07"
px = load_panel(VIS)
px = px.dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1], "VIS:", VIS)

# load macro
mac = {m: load_macro(m, VIS).reindex(px.index).ffill() for m in ['VIX','DXY','USDCNY','USDJPY','EURUSD']}

# Load library signal panels for correlation
lib = {}
for fid in ['beta_vix_60d_neg','sign_ewma_60d','vol_beta_spx_60d','mom_10d_skip5','mom_120d_skip5',
            'down_vol_ratio_20x120','beta_chi_60d','skew_20d_neg','xau_copper_cond_20d','vix_beta_cond_60x20']:
    try:
        d = json.load(open(f'factors/{fid}.json'))
        # skip loading full signal; we'll recompute common ones approximately
    except Exception:
        pass

def evalc(f, label, neg_ok=True):
    res = ic_analysis(f, px, horizon=10, label=label)
    icm = res['ic']; icir = res['icir']
    ic_signed = res.get('ic_signed')
    hit = res['ic_hit_ratio']
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840) and res['ic_signed'] is not None
    print(f"[{label}] n_ic={res['n_ic_dates']} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov_d8={res['coverage_dates_ge8']:.3f} turn={res['turnover_10d_rank']:.3f} GATE={'PASS' if gate else 'fail'}")
    return res

cands = {}

# A. DXY-regime momentum: 20d momentum only counted when DXY falling (weak-dollar = risk-on continuation)
dxy = mac['DXY']
dxy_fall = (dxy.shift(5) > dxy).astype(float)  # 1 when DXY declining
mom20 = px/px.shift(20) - 1
mom20_skip = mom20 - (px/px.shift(5)-1).shift(15).fillna(0)  # approx skip last 5
cands['weakdxy_mom20'] = mom20 * dxy_fall

# B. VIX level regime: momentum active only in low-VIX (risk-on) environment
vix = mac['VIX']
lowvix = (vix < vix.rolling(60).median()).astype(float)
cands['lowvix_mom20'] = mom20 * lowvix

# C. Dispersion of cross-sectional momentum: low dispersion -> trend continuation
mom10 = px/px.shift(10) - 1
cs_disp = mom10.std(axis=1)
cands['mom_disp_neg'] = -cs_disp.to_frame().reindex(columns=px.columns)
for c in cands['mom_disp_neg'].columns:
    cands['mom_disp_neg'][c] = -cs_disp.values

# D. 20d streak: fraction of up days (persistence signal)
up = (ret > 0).astype(float)
streak20 = up.rolling(20).mean()
cands['upfrac_20'] = streak20

# E. Recovery-from-drawdown momentum: distance from rolling 60d max
dd = px / px.rolling(60).max() - 1
cands['recover_60_cls'] = -dd  # closer to max = stronger recovery/trend

# F. Cross-sectional reversal of long-run losers (contrarian on 120d)
mom120 = px/px.shift(120) - 1
cands['mom120_contrarian'] = -mom120

# G. Corridor efficiency: |close-close trend| / total path proxy (trend quality)
hl = (px.pct_change().abs()).rolling(20).sum()
net = (px/px.shift(20)-1).abs()
eff = net / hl.replace(0, np.nan)
cands['trend_eff_20'] = eff

# H. CN10Y-US10Y spread beta (yield-curve signal): assets rising when CN-US spread widens
spread = mac['CN10Y'] if 'CN10Y' in mac else None
# approximate slope using index_data cn10y not in macro; use yield asset move instead
# Use US10Y level fall = duration/risk-off? test asset beta to US10Y change
us10y = px['US10Y']
usy_chg = (us10y.pct_change().rolling(20).mean())
usy_beta = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    usy_beta[a] = ret[a].rolling(60).corr(us10y.pct_change())
cands['us10y_chg_beta_60'] = usy_beta

# I. Cross-asset breadth momentum: mean of asset momentum relative to cross-section
cs_breadth = mom10.mean(axis=1)
cands['breadth_mom10'] = cs_breadth.to_frame().reindex(columns=px.columns)
for c in cands['breadth_mom10'].columns:
    cands['breadth_mom10'][c] = cs_breadth.values

for name, f in cands.items():
    evalc(f, name)