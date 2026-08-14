"""miner_3 re-validation of existing effective library factors (drift check) through 2034-04-26."""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner3_20340427_lib_utils import (load_all, close_panel, ret_panel, forward_ret,
                                       daily_spearman_ic, ic_stats, print_stats,
                                       coverage_stats, rank_turnover, WATCH)

data = load_all(days=4200)
px = close_panel(data)
ret = ret_panel(data)
fwd10 = forward_ret(px, 10)

print(f'panel dates: {px.index[0].date()} .. {px.index[-1].date()} rows={len(px)}')
print(f'assets loaded: {len(px.columns)}')

# ---- build library factor panels from formulas ----
def z(x):
    return (x - x.mean()) / x.std(ddof=0)

def roll_rank(x, w):
    return x.rolling(w, min_periods=w // 3).apply(lambda a: pd.Series(a).rank(pct=True).iloc[-1], raw=False)

factors = {}
# momentum family (skip-5 momentum = close t / close t-5 - 1 style)
for w in [10, 20, 60, 120, 180]:
    factors[f'mom_{w}d_skip5'] = px / px.shift(w + 5) - 1.0
factors['mom20_volproxy60'] = (px / px.shift(25) - 1.0) / ret.rolling(60).std()
factors['mom30_vol60'] = (px / px.shift(35) - 1.0) / ret.rolling(60).std()
factors['range_pos_252'] = ((px - px.rolling(252, min_periods=30).min()) /
                            (px.rolling(252, min_periods=30).max() - px.rolling(252, min_periods=30).min()))
factors['close_pos_20'] = ((px - px.rolling(20, min_periods=5).min()) /
                           (px.rolling(20, min_periods=5).max() - px.rolling(20, min_periods=5).min()))
factors['days_since_high_60'] = -(px.rolling(60, min_periods=20).apply(
    lambda a: np.argmax(a[::-1]) if len(a) else np.nan, raw=True))
factors['gain_loss_20'] = ret.clip(lower=0).rolling(20).sum() / (ret.clip(upper=0).rolling(20).sum().abs() + 1e-12)
factors['max_consec_gain_20'] = ret.apply(lambda c: c.gt(0).rolling(20).apply(
    lambda a: np.max(np.diff(np.r_[0, np.where(~a)[0], len(a)])) - 1 if (~a).any() else len(a), raw=True))
factors['max_consec_loss_20'] = ret.apply(lambda c: c.lt(0).rolling(20).apply(
    lambda a: np.max(np.diff(np.r_[0, np.where(~a)[0], len(a)])) - 1 if (~a).any() else len(a), raw=True))
factors['calmness_20'] = ret.apply(lambda c: (c.abs() < 0.5 * c.rolling(20).std()).rolling(20).mean())
factors['volcluster_60'] = ret.rolling(60).std() / ret.rolling(60).std().rolling(120).mean()
vol20 = ret.rolling(20).std()
vol60 = ret.rolling(60).std()
factors['vol_of_vol20x60'] = vol20 / vol60
factors['intraday_drift_20'] = px / px.shift(1) - 1.0  # placeholder overwritten below
# intraday drift: close-to-close vs open-to-close persistence (approx via open)
intra = pd.DataFrame({s: (data[s].set_index('date')['close'] / data[s].set_index('date')['open'] - 1.0)
                      for s in WATCH}).sort_index()
factors['intraday_drift_20'] = intra.rolling(20).mean() * np.sqrt(20)
# skewness
factors['ret_skew_10'] = ret.rolling(10).skew()

# cross-asset beta factors
spx_ret = ret['SPX']
def downbeta(r, w=60):
    mask = spx_ret < spx_ret.rolling(w).median()
    cov = (r * spx_ret).rolling(w).sum() / mask.rolling(w).sum().clip(lower=1)
    var = (spx_ret ** 2 * mask).rolling(w).sum() / mask.rolling(w).sum().clip(lower=1)
    return cov / (var + 1e-12)
factors['downbeta_spx_60'] = downbeta(ret, 60)
factors['lagbeta_spx_60'] = ret.apply(lambda c: c.shift(1).rolling(60).cov(spx_ret.shift(1)) /
                                      spx_ret.shift(1).rolling(60).var())
factors['spx_corr60'] = ret.apply(lambda c: c.rolling(60).corr(spx_ret))
# macro-conditional betas
dxy = pd.read_csv('../persistent/index_data/DXY.csv'); dxy['date'] = pd.to_datetime(dxy['date'])
dxy = dxy[dxy['date'] <= px.index[-1]].set_index('date')['close']; dxy_r = dxy.pct_change()
dxy_r = dxy_r.reindex(px.index).ffill()
up_dxy = dxy_r > 0
factors['dxy_beta_cond_60x20'] = ret.apply(
    lambda c: (c * dxy_r).rolling(60).sum() / up_dxy.rolling(60).sum().clip(lower=1) /
    ((dxy_r ** 2 * up_dxy).rolling(60).sum() / up_dxy.rolling(60).sum().clip(lower=1) + 1e-12))
usdjpy = pd.read_csv('../persistent/index_data/USDJPY.csv'); usdjpy['date'] = pd.to_datetime(usdjpy['date'])
usdjpy = usdjpy[usdjpy['date'] <= px.index[-1]].set_index('date')['close']; usdjpy_r = usdjpy.pct_change().reindex(px.index).ffill()
up_jpy = usdjpy_r > 0
factors['usdjpy_beta_cond_120x60'] = ret.apply(
    lambda c: (c * usdjpy_r).rolling(120).sum() / up_jpy.rolling(120).sum().clip(lower=1) /
    ((usdjpy_r ** 2 * up_jpy).rolling(120).sum() / up_jpy.rolling(120).sum().clip(lower=1) + 1e-12))
vix = pd.read_csv('../persistent/index_data/VIX.csv'); vix['date'] = pd.to_datetime(vix['date'])
vix = vix[vix['date'] <= px.index[-1]].set_index('date')['close']; vix_r = vix.pct_change().reindex(px.index).ffill()
up_vix = vix_r > 0
factors['vix_beta_cond_60x20'] = ret.apply(
    lambda c: (c * vix_r).rolling(60).sum() / up_vix.rolling(60).sum().clip(lower=1) /
    ((vix_r ** 2 * up_vix).rolling(60).sum() / up_vix.rolling(60).sum().clip(lower=1) + 1e-12))

print(f'\n=== LIBRARY DRIFT CHECK (10d horizon, through 2034-04-26) ===')
full_stats = {}
for name, f in factors.items():
    m = ic_stats(daily_spearman_ic(f, fwd10))
    if m is None:
        continue
    full_stats[name] = m
    cov = coverage_stats(f)
    to = rank_turnover(f)
    # recent 2y window
    f_recent = f[f.index >= '2032-05-01']
    fwd_recent = fwd10.reindex(f_recent.index)
    m_r = ic_stats(daily_spearman_ic(f_recent, fwd_recent))
    r_str = f" recent2y ic={m_r['ic']:.4f} icir={m_r['icir']:.3f}" if m_r else ' recent2y NONE'
    flag = 'PASS' if (abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084) else 'FAIL'
    flag_r = 'PASS' if (m_r and abs(m_r['ic']) >= 0.007 and abs(m_r['icir']) >= 0.084) else 'fail'
    print(f"{name:24s} full ic={m['ic']:+.4f} icir={m['icir']:+.3f} hit={m['ic_hit_ratio']:.2f} "
          f"n={m['n_ic_dates']:4d} cov={cov['coverage_dates_ge8']:.2f} to={to:.3f} | {flag} |{r_str} |{flag_r}")
