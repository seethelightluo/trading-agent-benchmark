"""miner_1: revalidate existing ensemble factors + explore new candidates.
Visible through 2035-07-18. Current cycle 2035-07-19. No lookahead.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = pd.Timestamp('2035-07-18')
SD = Path('../persistent/stock_data')
ID = Path('../persistent/index_data')
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load_ohlcv(assets=None, end=VISIBLE_END):
    assets = assets or ASSETS
    out = {}
    for a in assets:
        f = SD / f'{a}.csv'
        if not f.exists():
            f = ID / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
        df = df[df['date'] <= end].set_index('date')
        out[a] = df
    return out


def load_macro(name, end=VISIBLE_END):
    df = pd.read_csv(ID / f'{name}.csv', parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= end].set_index('date')
    return df['close'].astype(float)


dob = load_ohlcv(ASSETS)
close = pd.DataFrame({a: dob[a]['close'].astype(float) for a in ASSETS})
high = pd.DataFrame({a: dob[a]['high'].astype(float) for a in ASSETS})
low = pd.DataFrame({a: dob[a]['low'].astype(float) for a in ASSETS})
vol = pd.DataFrame({a: (dob[a]['volume'].astype(float) if 'volume' in dob[a] else np.nan) for a in ASSETS}).reindex(close.index)
rets = close.pct_change().dropna()

dVIX = load_macro('VIX').reindex(close.index)
dDXY = load_macro('DXY').reindex(close.index)
dCNY = load_macro('USDCNY').reindex(close.index)
dJPY = load_macro('USDJPY').reindex(close.index)


def compute_ic(fv, fwd, start=None, min_dates=30, min_assets=8, flip=False):
    if start is not None:
        fv = fv[fv.index >= pd.Timestamp(start)]
    common = sorted(set(fv.index) & set(fwd.index))
    ics, okd = [], 0
    for d in common:
        f = fv.loc[d]; r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= min_assets:
            okd += 1
            x = f[m].rank().values; y = r[m].rank().values
            if np.std(x) > 0 and np.std(y) > 0:
                ics.append(np.corrcoef(x, y)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return dict(IC=0., ICIR=0., n=len(ics), okd=okd, hit=0., cov=0.)
    mu = ics.mean(); sd = ics.std()
    if flip:
        mu = -mu
    cov = float(fv.notna().mean().mean())
    return dict(IC=float(mu), ICIR=float(mu / sd * np.sqrt(len(ics)) if sd > 0 else 0),
                n=len(ics), okd=okd, hit=float((ics > 0).mean()), cov=cov)


def turnover(fv):
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1] / 2)).fillna(0)
    return float((s.diff() != 0).mean().mean())


fwd5 = rets.shift(-5).rolling(5).mean()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd20 = rets.shift(-20).rolling(20).mean()

RECENT = '2032-01-01'
FULL = '2022-01-01'
print(f"Panel {close.shape[0]}x{close.shape[1]} end {VISIBLE_END.date()}", flush=True)


def calc_beta(macro_ret, win=60):
    m = macro_ret.name
    out = {}
    for a in ASSETS:
        r = rets[a]
        mm = pd.concat([r, macro_ret], axis=1).dropna()
        out[a] = mm[r].rolling(win, min_periods=40).cov(mm[m]) / mm[m].rolling(win, min_periods=40).var()
    return pd.DataFrame(out).reindex(close.index)


def calc_corr_change(macro_ret, sw=20, lw=60):
    m = macro_ret.name
    out = {}
    for a in ASSETS:
        r = rets[a]
        mm = pd.concat([r, macro_ret], axis=1).dropna()
        out[a] = mm[r].rolling(sw).corr(mm[m]) - mm[r].rolling(lw).corr(mm[m])
    return pd.DataFrame(out).reindex(close.index)


def calc_mom(skip, lb):
    return close.shift(skip) / close.shift(skip + lb) - 1


def report(name, fv, start=None, flip=False):
    a = compute_ic(fv, fwd10, start=start, flip=flip)
    b = compute_ic(fv, fwd5, start=start, flip=flip)
    c = compute_ic(fv, fwd20, start=start, flip=flip)
    ok = abs(a['IC']) >= 0.0070 and abs(a['ICIR']) >= 0.084
    print(f"[{'OK' if ok else '--'}] {name}: IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} "
          f"n={a['n']} okd={a['okd']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} "
          f"| [5]{b['IC']:+.3f} [20]{c['IC']:+.3f}", flush=True)
    return a, ok


vixr = dVIX.pct_change().rename('vixr')
dxyxr = dDXY.pct_change().rename('dxyx')
cnyr = dCNY.pct_change().rename('cnyr')
wti = close['WTI'].pct_change().rename('wti')
xau = close['XAU'].pct_change().rename('xau')
kaufman = (close.diff(20).abs() / close.diff().abs().rolling(20).sum().replace(0, np.nan))
bw20 = 4 * close.rolling(20).std() / close.rolling(20).mean()
skew20 = rets.rolling(20).skew()
volz20 = ((vol - vol.rolling(20).mean()) / vol.rolling(20).std()).reindex(close.index)
mom120 = calc_mom(5, 120)
mom10 = calc_mom(5, 10)

print("=== EXISTING ENSEMBLE (RECENT 2032+) ===", flush=True)
report('beta_VIX_60', calc_beta(vixr, 60), RECENT, flip=True)
report('kaufman_eff_20d', kaufman, RECENT)
report('mom_120d_skip5', mom120, RECENT)
report('mom_10d_skip5', mom10, RECENT)
report('bb_width_20d', bw20, RECENT)
report('cny_beta_60', calc_beta(cnyr, 60), RECENT)
report('vol_z_20d', volz20, RECENT)
report('dxy_corr_change_20_60', calc_corr_change(dxyx, 20, 60), RECENT)
report('skew_20d', skew20, RECENT)

print("=== EXISTING ENSEMBLE (FULL 2022+) ===", flush=True)
report('beta_VIX_60', calc_beta(vixr, 60), FULL, flip=True)
report('mom_