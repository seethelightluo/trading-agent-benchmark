"""miner_1 2028-09-07: re-validate the 10 ensemble factors on extended windows.

Windows:
  W0 warm-up   : 2020-01-01 .. 2026-07-15 (baseline vs recorded admission metrics)
  W1 full      : 2020-01-01 .. 2028-09-06 (visible through previous trading day)
  W2 recent 12m: 2027-09-06 .. 2028-09-06
  W3 recent 6m : 2028-03-07 .. 2028-09-06

Admission gate: |IC|>=0.0070 and |ICIR|>=0.0840 (10d horizon, daily cross-sectional Spearman).
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, forward_returns, VAL_START, VAL_END

# ---------------------------------------------------------------- data
prices = load_prices(days=2500)
print('assets loaded:', len(prices))
print('visible price horizon:', min(d.index.max() for d in prices.values()), '..', max(d.index.max() for d in prices.values()))
vix = load_index('VIX', prices=prices)
print('vix horizon:', vix.index.min(), '..', vix.index.max())

# ---------------------------------------------------------------- factor impls
def _beta(a, b, win, min_obs=60):
    """rolling beta of a on b"""
    m = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    cov = m['a'].rolling(win, min_periods=min_obs).cov(m['b'])
    var = m['b'].rolling(win, min_periods=min_obs).var()
    return (cov / var).reindex(a.index)

def f_cn10y(df, s):
    r = df['close'].pct_change()
    cn = prices['CN10Y']['close']
    dcn = cn.diff()
    b = _beta(r, dcn, 60)
    return b

def f_vol_adj_mom(df, s):
    mom = df['close'].shift(5) / df['close'].shift(25) - 1.0
    vol = df['close'].pct_change().rolling(60).std()
    return mom / vol

def f_hs300_beta(df, s):
    r = df['close'].pct_change()
    rh = prices['000300.SH']['close'].pct_change()
    return _beta(r, rh, 60, min_obs=70)

def f_comm_basket(df, s):
    r = df['close'].pct_change()
    bk = np.mean([prices[x]['close'].pct_change() for x in ['XAU', 'COPPER', 'WTI']], axis=0)
    return _beta(r, bk, 60)

def f_hilo_vol(df, s):
    hi = df['close'].rolling(20).max()
    lo = df['close'].rolling(20).min()
    vol = df['close'].pct_change().rolling(20).std()
    return (hi - lo) / df['close'] / vol

def f_vov(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

def f_vix_cond(df, s):
    r = df['close'].pct_change()
    vr = vix['close'].pct_change()
    b = _beta(r, vr, 60)
    chg = vix['close'] / vix['close'].shift(20) - 1.0
    return (-b * chg).reindex(df.index)

def f_vol_regime(df, s):
    rv = df['close'].pct_change().rolling(20).std()
    med = rv.rolling(60).median()
    state = (rv > med).astype(float)
    switch = state.diff().abs()
    return switch.rolling(60).mean()

def f_intraday_skew(df, s):
    x = df['close'] / df['open'] - 1.0
    return x.rolling(20).skew()

def f_dd_duration(df, s):
    hi = df['close'].rolling(120, min_periods=60).max()
    since = (df.index.to_series().rolling(120, min_periods=60).apply(
        lambda idx: (idx[-1] - idx[idx == idx.max()][0]).days if idx.max() == idx[-1] else (
            (idx[-1] - df['close'].loc[idx].idxmax()).days), raw=False))
    # simpler: days since the rolling window's argmax
    days_since = pd.Series(np.nan, index=df.index)
    roll_idx = df['close'].rolling(120, min_periods=60).apply(lambda x: x.argmax(), raw=True)
    for i in range(59, len(df)):
        w = df['close'].iloc[i-119:i+1]
        if len(w.dropna()) < 60:
            continue
        argmax_pos = int(w.values.argmax())
        days_since.iloc[i] = argmax_pos  # 0 = at high
    return np.log1p(days_since)

def f_dd_duration_resid(df, s):
    dd = f_dd_duration(df, s)
    mom = df['close'].shift(5) / df['close'].shift(125) - 1.0
    out = pd.Series(np.nan, index=df.index)
    for d in dd.index:
        if np.isfinite(dd.loc[d]) and np.isfinite(mom.loc[d]):
            out.loc[d] = dd.loc[d]
    # cross-sectional orthogonalization done in panel stage (needs all assets)
    return out

FACTORS = {
    'cn10y_beta_60': f_cn10y,
    'vol_adj_mom_20_60': f_vol_adj_mom,
    'hs300_beta_60': f_hs300_beta,
    'comm_basket_beta_60': f_comm_basket,
    'hilo_vol_ratio_20': f_hilo_vol,
    'vol_of_vol20x60': f_vov,
    'vix_beta_cond_60x20': f_vix_cond,
    'vol_regime_switch_20x60': f_vol_regime,
    'intraday_ret_skew_20': f_intraday_skew,
    'dd_duration_120_resid': f_dd_duration_resid,
}

# ---------------------------------------------------------------- panel + ortho
def build_panels():
    panels = {}
    for fid, fn in FACTORS.items():
        cols = {}
        for s, df in prices.items():
            try:
                ser = fn(df, s)
                if ser is not None and len(ser):
                    cols[s] = ser.astype(float)
            except Exception as e:
                print(f'  {fid} {s} ERR {e}')
        panels[fid] = pd.DataFrame(cols).sort_index()
    # orthogonalize dd_duration against mom120 per date
    mom120 = pd.DataFrame({s: prices[s]['close'].shift(5) / prices[s]['close'].shift(125) - 1.0
                           for s in prices}).sort_index()
    dd = panels['dd_duration_120_resid']
    z_mom = mom120.sub(mom120.mean(axis=1), axis=0).div(mom120.std(axis=1), axis=0)
    resid = dd.copy()
    for d in dd.index:
        x = dd.loc[d]; z = z_mom.loc[d]
        m = x.notna() & z.notna() & np.isfinite(x) & np.isfinite(z)
        if m.sum() >= 8:
            xv = x[m].values; zv = z[m].values
            b = np.cov(zv, xv)[0, 1] / (np.var(zv) + 1e-12)
            resid.loc[d, m] = xv - b * zv
    panels['dd_duration_120_resid'] = resid
    return panels

# ---------------------------------------------------------------- validation
def validate_window(panel, fwd, start, end, min_valid=8):
    ic10 = []
    common = panel.index.intersection(fwd.index)
    for d in common:
        if d < start or d > end:
            continue
        x = panel.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            ic10.append(x[m].rank().corr(y[m].rank()))
    ic10 = pd.Series(ic10)
    if len(ic10) < 60:
        return None
    ic_mean = float(ic10.mean())
    ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    return {'ic': ic_mean, 'icir': icir, 'hit': hit, 'n': int(len(ic10))}

WINDOWS = {
    'W0_warmup_2020_2026h1': (pd.Timestamp('2020-01-01'), pd.Timestamp('2026-07-15')),
    'W1_full_2020_2028': (pd.Timestamp('2020-01-01'), pd.Timestamp('2028-09-06')),
    'W2_12m_2027_2028': (pd.Timestamp('2027-09-06'), pd.Timestamp('2028-09-06')),
    'W3_6m_2028': (pd.Timestamp('2028-03-07'), pd.Timestamp('2028-09-06')),
}

def main():
    panels = build_panels()
    fwd = forward_returns(prices, 10)
    print('\n================ RE-VALIDATION (10d horizon) ================')
    summary = {}
    for fid in FACTORS:
        panel = panels[fid]
        row = {}
        for wname, (s, e) in WINDOWS.items():
            r = validate_window(panel, fwd, s, e)
            row[wname] = r
        summary[fid] = row
        line = [f'{fid:26s}']
        for wname in WINDOWS:
            r = row[wname]
            if r is None:
                line.append(f'{wname}: NA')
            else:
                flag = 'PASS' if (abs(r['ic']) >= 0.007 and abs(r['icir']) >= 0.084) else 'fail'
                line.append(f'{wname}: IC={r["ic"]:+.4f} ICIR={r["icir"]:+.3f} hit={r["hit"]:.2f} n={r["n"]} [{flag}]')
        print('\n'.join(['  ' + l for l in line]))
    # recorded warm-up baselines for sanity
    print('\nRecorded admission baselines (warm-up):')
    rec = {
        'cn10y_beta_60': (-0.0622, -0.187), 'vol_adj_mom_20_60': (0.0582, 0.175),
        'hs300_beta_60': (-0.0449, -0.125), 'comm_basket_beta_60': (0.0428, 0.122),
        'hilo_vol_ratio_20': (0.0418, 0.129), 'vol_of_vol20x60': (0.0424, 0.121),
        'vix_beta_cond_60x20': (-0.0382, -0.093), 'vol_regime_switch_20x60': (0.0375, 0.131),
        'intraday_ret_skew_20': (0.0395, 0.133), 'dd_duration_120_resid': (-0.0330, -0.112),
    }
    for fid, (ic, icir) in rec.items():
        w0 = summary[fid]['W0_warmup_2020_2026h1']
        if w0:
            print(f'  {fid:26s} recorded IC={ic:+.4f}/{icir:+.3f}  recomputed IC={w0["ic"]:+.4f}/{w0["icir"]:+.3f}')

if __name__ == '__main__':
    main()
