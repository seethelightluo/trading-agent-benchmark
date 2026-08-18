"""miner_3 exploration 2030-04-04 (optimized): batch screen of novel factor candidates."""
import sys, time
sys.path.insert(0, "scripts")
from miner3_20300404_common import *

t0 = time.time()
px, rets = load_asset_panel()
print(f"Panel: {px.shape[0]} dates x {px.shape[1]} assets, {px.index[0].date()}..{px.index[-1].date()} [{time.time()-t0:.1f}s]")

t0 = time.time()
fwd = build_fwd_returns(rets, horizons=(5, 10, 20))
print(f"fwd rets [{time.time()-t0:.1f}s]")

def roll_beta(ret_asset, ret_ref, win, minp):
    cov = ret_asset.rolling(win, min_periods=minp).cov(ret_ref)
    var = ret_ref.rolling(win, min_periods=minp).var()
    return cov / var

t0 = time.time()
factors = {}
r_dxy = rets["DXY"]
factors["dxy_beta_60d"] = roll_beta(rets[TRADABLE], r_dxy, 60, 30)
r_jpy = rets["USDJPY"]
factors["usdjpy_beta_60d"] = roll_beta(rets[TRADABLE], r_jpy, 60, 30)
r_vix = rets["VIX"]
factors["vix_chg_sens_20d"] = roll_beta(rets[TRADABLE], r_vix, 20, 10)
hi20 = px[TRADABLE].rolling(20, min_periods=10).max()
lo20 = px[TRADABLE].rolling(20, min_periods=10).min()
factors["range_pos_20d"] = (px[TRADABLE] - lo20) / (hi20 - lo20)
hi10 = px[TRADABLE].rolling(10, min_periods=5).max()
lo10 = px[TRADABLE].rolling(10, min_periods=5).min()
factors["range_pos_10d"] = (px[TRADABLE] - lo10) / (hi10 - lo10)
dist = (px[TRADABLE] - px[TRADABLE].shift(20)).abs()
path = rets[TRADABLE].abs().rolling(20, min_periods=10).sum()
factors["kaufman_eff_20d"] = dist / path
# autocorr 5d via vectorized rolling dot
r = rets[TRADABLE]
lag = r.shift(1)
num = (r * lag).rolling(5, min_periods=4).mean() - r.rolling(5, min_periods=4).mean() * lag.rolling(5, min_periods=4).mean()
den = r.rolling(5, min_periods=4).std() * lag.rolling(5, min_periods=4).std()
factors["autocc_5d"] = num / den
roll_max = px[TRADABLE].rolling(60, min_periods=30).max()
factors["maxdd_60d_neg"] = (px[TRADABLE] / roll_max - 1.0)
factors["relstr_20d_vs_spx"] = px[TRADABLE].pct_change(20) - px["SPX"].pct_change(20)
factors["relstr_60d_vs_spx"] = px[TRADABLE].pct_change(60) - px["SPX"].pct_change(60)
vol5 = rets[TRADABLE].rolling(5, min_periods=3).std()
vol60 = rets[TRADABLE].rolling(60, min_periods=30).std()
factors["vol_ratio_5x60"] = vol5 / vol60
factors["tail_risk_20d"] = rets[TRADABLE].rolling(20, min_periods=10).kurt()
r_wti = rets["WTI"]
factors["wti_beta_60d"] = roll_beta(rets[TRADABLE], r_wti, 60, 30)
r_xau = rets["XAU"]
factors["xau_beta_60d"] = roll_beta(rets[TRADABLE], r_xau, 60, 30)
print(f"factors computed [{time.time()-t0:.1f}s]")

t0 = time.time()
for name, sig in factors.items():
    res = evaluate_factor(sig, fwd[10], fwd[5], fwd[20], label=name)
    h10 = res["h10"]
    if h10 is None:
        print(f"{name:22s} NO VALID IC")
        continue
    rec = res.get("recent_h10", {})
    last6 = res.get("last6m_h10", {})
    print(f"{name:22s} IC10={h10['ic']:+.4f} ICIR10={h10['icir']:+.4f} hit={h10['hit']:.3f} n={h10['n_dates']:4d} "
          f"cov={res['coverage_asset_days']:.2f}/{res['coverage_dates_ge8']:.2f} | "
          f"IC5={res['h5']['ic']:+.4f} IC20={res['h20']['ic']:+.4f} | "
          f"rec28+ IC={rec.get('ic', float('nan')):+.4f} ICIR={rec.get('icir', float('nan')):+.4f} | "
          f"last6m IC={last6.get('ic', float('nan')):+.4f} ICIR={last6.get('icir', float('nan')):+.4f}")
print(f"eval done [{time.time()-t0:.1f}s]")
