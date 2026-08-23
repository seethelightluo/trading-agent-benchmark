"""miner_3 2035-11-07: revalidate effective library through visible 2035-11-06 (fixed)."""
import pandas as pd, numpy as np, math
import sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import TRADABLE, rank_ic_series, align_fwd_returns

VIS = '2035-11-06'
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= pd.Timestamp(VIS)].sort_values('date')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    closes[sym] = df.set_index('date')['close']
px = pd.DataFrame(closes).sort_index().ffill()
px = px[px.index >= '2021-01-01']
ret = px.pct_change()
vix = pd.read_csv('../persistent/index_data/VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= pd.Timestamp(VIS)].set_index('date')['close'].reindex(px.index).ffill()
vixr = vix.pct_change()
print('panel', px.shape, px.index.min().date(), '->', px.index.max().date(), flush=True)


def wrap(f):
    if isinstance(f, pd.Series):
        return pd.DataFrame({c: f for c in px.columns})
    return f


def sig_mom10(): return px.shift(5) / px.shift(15) - 1.0
def sig_mom120(): return px.shift(5) / px.shift(125) - 1.0
def sig_vov(): return ret.rolling(20).std().rolling(60).std()
def sig_lowvol(): return -ret.rolling(20).std()
def sig_betavix_neg():
    b = ret.rolling(60).cov(vixr) / vixr.rolling(60).var(); return -b
def sig_vixcond():
    b = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
    return -b * (vix / vix.shift(20) - 1.0)
def sig_dvr(): return wrap(ret.clip(upper=0).rolling(20).std() / ret.rolling(120).std())
def sig_betacn10():
    r10 = px['CN10Y'].pct_change(); return ret.rolling(60).cov(r10) / r10.rolling(60).var()
def sig_betachi():
    rhi = px['HSI'].pct_change(); return ret.rolling(60).cov(rhi) / rhi.rolling(60).var()
def sig_corr10y():
    r10 = px['US10Y'].pct_change(); return ret.rolling(60).corr(r10)
def sig_skew(): return -ret.rolling(20).skew()
def sig_vovchg():
    vv = ret.rolling(20).std(); return vv.diff(20) / vv.rolling(20).mean()
def sig_xaucop():
    cond = ((px['XAU'].pct_change(20) > 0) & (px['COPPER'].pct_change(20) > 0)).astype(float)
    return pd.DataFrame({c: -cond for c in px.columns})
def sig_volbeta():
    vv = ret.rolling(20).std(); spxv = ret['SPX'].rolling(20).std()
    return vv.rolling(60).cov(spxv) / spxv.rolling(60).var()
def sig_signewma():
    return (px / px.ewm(span=60).mean() - 1.0).apply(np.sign)

factors = {'mom_10d_skip5': sig_mom10, 'mom_120d_skip5': sig_mom120, 'vol_of_vol20x60': sig_vov,
           'low_vol_20d': sig_lowvol, 'beta_vix_60d_neg': sig_betavix_neg, 'vix_beta_cond_60x20': sig_vixcond,
           'down_vol_ratio_20x120': sig_dvr, 'beta_cn10y_60d': sig_betacn10, 'beta_chi_60d': sig_betachi,
           'corr_us10y_60d': sig_corr10y, 'skew_20d_neg': sig_skew, 'vol_of_vol_chg_20d': sig_vovchg,
           'xau_copper_cond_20d': sig_xaucop, 'vol_beta_spx_60d': sig_volbeta, 'sign_ewma_60d': sig_signewma}


def summarize(ic):
    icm = float(ic.mean())
    icstd = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = icm / icstd if icstd and math.isfinite(icstd) and icstd > 0 else np.nan
    return icm, icir, float((ic > 0).mean()), len(ic)


res = []
for name, fn in factors.items():
    try:
        f = fn()
        ic = rank_ic_series(f, align_fwd_returns(px, 10))
        icm, icir, hit, n = summarize(ic)
        recent = ic[ic.index >= '2033-01-01']
        ricm, ricir, _, _ = summarize(recent) if len(recent) > 2 else (np.nan, np.nan, np.nan, 0)
        cov = float(f.notna().mean().mean())
        gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
        flag = 'PASS' if gate else 'fail'
        print(f'{name:22s} IC10={icm:+.4f} ICIR10={icir:+.4f} hit={hit:.3f} n={n} '
              f'recentIC={ricm:+.4f} ricir={ricir:+.4f} cov={cov:.3f} {flag}', flush=True)
        res.append(dict(factor=name, ic=round(icm, 4), icir=round(icir, 4) if math.isfinite(icir) else None,
                        hit=round(hit, 3), n=n, recent_ic=round(ricm, 4) if ricm == ricm else None,
                        recent_icir=round(ricir, 4) if ricir == ricir else None, cov=round(cov, 3), gate=flag))
    except Exception as e:
        print(f'{name} ERROR {e}', flush=True)
import json
json.dump(res, open('scripts/miner3_20351107_reval_fixed.json', 'w'), indent=1)
print('saved', flush=True)