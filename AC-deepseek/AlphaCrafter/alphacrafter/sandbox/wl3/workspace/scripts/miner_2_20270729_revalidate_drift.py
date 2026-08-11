"""Drift re-validation of all currently EFFECTIVE library factors (FAST version).

Vectorized daily cross-sectional Spearman IC.
"""
import sys, json, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import load_prices, load_index, WATCHLIST, factor_to_panel

t0 = time.time()
prices = load_prices(days=4000)
vix = load_index('VIX', days=4000, prices=prices)
dxy = load_index('DXY', days=4000, prices=prices)
eur = load_index('EURUSD', days=4000, prices=prices)
print(f"data loaded {time.time()-t0:.1f}s")

R = {s: df['close'].pct_change() for s, df in prices.items()}
RV20 = {s: df['close'].pct_change().rolling(20).std() for s, df in prices.items()}
RV60 = {s: df['close'].pct_change().rolling(60).std() for s, df in prices.items()}


def f_cn10y(df, s): return R[s].rolling(60).cov(R['CN10Y']) / R['CN10Y'].rolling(60).var()
def f_comm(df, s):
    ew = (R['XAU'] + R['COPPER'] + R['WTI']) / 3.0
    return R[s].rolling(60).cov(ew) / ew.rolling(60).var()
def f_copgold(df, s):
    x = R['COPPER'] - R['XAU']
    return R[s].rolling(20).cov(x) / x.rolling(20).var()
def f_dd_dur(df, s):
    mom = df['close'].shift(5) / df['close'].shift(125) - 1.0
    dd = np.log1p((df['close'].rolling(120).max() - df['close']) / df['close'].rolling(120).max())
    z = (mom - mom.rolling(120).mean()) / mom.rolling(120).std()
    b = R[s].rolling(120).cov(R['SPX']) / R['SPX'].rolling(120).var()
    return dd - b * z
def f_down(df, s):
    r = R[s]; m = R['SPX']
    d = pd.concat([r.rename('r'), m.rename('m')], axis=1, sort=True).dropna()
    sub = d[d['m'] < 0]
    if len(sub) < 30: return pd.Series(np.nan, index=d.index)
    return (sub['r'].rolling(60).cov(sub['m']) / sub['m'].rolling(60).var()).reindex(d.index)
def f_dxy(df, s):
    vr = dxy['close'].pct_change()
    return (R[s].rolling(60).cov(vr) / vr.rolling(60).var()) * (dxy['close'] / dxy['close'].shift(20) - 1.0)
def f_eur(df, s):
    vr = eur['close'].pct_change()
    return (R[s].rolling(60).cov(vr) / vr.rolling(60).var()) * (eur['close'] / eur['close'].shift(20) - 1.0)
def f_hlpos(df, s):
    hi = df['high'].rolling(60).max(); lo = df['low'].rolling(60).min()
    return (df['close'] - lo) / (hi - lo)
def f_hlvr(df, s):
    return ((df['close'].rolling(20).max() - df['close'].rolling(20).min()) / df['close']) / RV20[s]
def f_hs300(df, s): return R[s].rolling(60).cov(R['000300.SH']) / R['000300.SH'].rolling(60).var()
def f_intraday_skew(df, s): return (df['close'] / df['open'] - 1.0).rolling(20).skew()
def f_momaccel(df, s):
    return (df['close'].shift(5) / df['close'].shift(65) - 1.0) - (df['close'].shift(5) / df['close'].shift(125) - 1.0)
def f_range_skew(df, s): return ((df['high'] - df['low']) / df['close']).rolling(20).skew()
def f_signpersist(df, s):
    r = df['close'].pct_change()
    return (np.sign(r) == np.sign(r.shift(1))).astype(float).rolling(20).mean()
def f_spx(df, s): return R[s].rolling(60).cov(R['SPX']) / R['SPX'].rolling(60).var()
def f_streak(df, s):
    r = df['close'].pct_change()
    up = (r > 0).astype(int); dn = (r < 0).astype(int)
    ups = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    dns = dn * (dn.groupby((dn != dn.shift()).cumsum()).cumcount() + 1)
    return ((ups - dns).rolling(60).max() / 60.0)
def f_vixcond(df, s):
    vr = vix['close'].pct_change()
    return -(R[s].rolling(60).cov(vr) / vr.rolling(60).var()) * (vix['close'] / vix['close'].shift(20) - 1.0)
def f_vam(df, s): return (df['close'].shift(5) / df['close'].shift(25) - 1.0) / RV60[s]
def f_vov(df, s): return RV20[s].rolling(60).std()
def f_volreg(df, s):
    med = RV20[s].rolling(60).median()
    return ((RV20[s] > med).astype(float).diff().ne(0).astype(float).rolling(60).mean())

FACTORS = {
    'cn10y_beta_60': f_cn10y, 'comm_basket_beta_60': f_comm, 'copper_gold_beta_20': f_copgold,
    'dd_duration_120_resid': f_dd_dur, 'down_beta_60': f_down, 'dxy_beta_cond_60x20': f_dxy,
    'eurusd_beta_cond_60x20': f_eur, 'hilo_pos_60': f_hlpos, 'hilo_vol_ratio_20': f_hlvr,
    'hs300_beta_60': f_hs300, 'intraday_ret_skew_20': f_intraday_skew, 'mom_accel_60_120': f_momaccel,
    'range_skew_20': f_range_skew, 'sign_persist_20': f_signpersist, 'spx_beta_60': f_spx,
    'streak_60': f_streak, 'vix_beta_cond_60x20': f_vixcond, 'vol_adj_mom_20_60': f_vam,
    'vol_of_vol20x60': f_vov, 'vol_regime_switch_20x60': f_volreg,
}


def fast_rank_ic(fac, fwd, min_valid=8):
    """Vectorized daily cross-sectional Spearman IC via ranks."""
    # rank each column within each row (per date)
    fr = fac.rank(axis=1)
    yr = fwd.rank(axis=1)
    valid = fac.notna() & fwd.notna() & np.isfinite(fac) & np.isfinite(fwd)
    n = valid.sum(axis=1)
    ok = n >= min_valid
    # mean-center ranks on valid entries
    fr_m = fr.where(valid).sub(fr.where(valid).mean(axis=1), axis=0)
    yr_m = yr.where(valid).sub(yr.where(valid).mean(axis=1), axis=0)
    cov = (fr_m * yr_m).sum(axis=1) / (n - 1)
    sdf = fr_m.pow(2).sum(axis=1) / (n - 1)
    sdy = yr_m.pow(2).sum(axis=1) / (n - 1)
    ic = cov / np.sqrt(sdf * sdy)
    return ic[ok].replace([np.inf, -np.inf], np.nan).dropna()


def ic_stats(panel, prices, start, end, h=10, min_valid=8):
    fwd = pd.DataFrame({s: df['close'].shift(-h) / df['close'] - 1.0 for s, df in prices.items()})
    fwd = fwd.reindex(panel.index)
    ic = fast_rank_ic(panel, fwd, min_valid)
    ic = ic[(ic.index >= start) & (ic.index <= end)]
    if len(ic) < 60: return None
    mu = float(ic.mean()); sd = float(ic.std(ddof=1))
    return dict(ic=mu, icir=mu / sd if sd > 0 else 0.0, n=len(ic))


WARM = (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-07-15'))
ONLN = (pd.Timestamp('2026-07-16'), pd.Timestamp('2027-07-28'))
FULL = (pd.Timestamp('2020-01-01'), pd.Timestamp('2027-07-28'))

print(f"{'factor':24s} {'warm_ic':>8s} {'warm_icir':>9s} | {'full_ic':>8s} {'full_icir':>9s} | {'onln_ic':>8s} {'onln_icir':>9s}")
out = {}
for fid, fn in FACTORS.items():
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid:24s} EMPTY"); continue
    w = ic_stats(panel, prices, *WARM); f = ic_stats(panel, prices, *FULL); o = ic_stats(panel, prices, *ONLN)
    out[fid] = {'warm': w, 'full': f, 'online': o}
    ww = f"{w['ic']:+.4f}/{w['icir']:+.3f}" if w else "n/a"
    ff = f"{f['ic']:+.4f}/{f['icir']:+.3f}" if f else "n/a"
    oo = f"{o['ic']:+.4f}/{o['icir']:+.3f}" if o else "n/a"
    flag = ""
    if w and o:
        gate = abs(o['ic']) >= 0.007 and abs(o['icir']) >= 0.084
        flag = "OK" if gate else ("WEAK" if abs(o['ic']) > 0 else "FLIP")
    print(f"{fid:24s} {ww:>18s} | {ff:>18s} | {oo:>18s}  {flag}")

json.dump(out, open('scripts/miner_2_20270729_reval_drift.json', 'w'), indent=1, default=str)
print(f"done {time.time()-t0:.1f}s -> scripts/miner_2_20270729_reval_drift.json")
