"""Screener fresh IC refresh through 2035-06-14 (visible_through). No lookahead."""
import pandas as pd, numpy as np, json, glob

END = '2035-06-14'
SYMS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# ---------- load close panel ----------
close = {}
for s in SYMS:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= END].set_index('date')
    close[s] = df['close']
px = pd.DataFrame(close).sort_index()
px = px[~px.index.duplicated(keep='last')]
print('panel', px.shape, px.index[0].date(), '->', px.index[-1].date())

# VIX for macro factor
vix = pd.read_csv('../persistent/index_data/VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= END].set_index('date')['close']
vix = vix[~vix.index.duplicated(keep='last')]

# ---------- factor signals ----------
def sig_nclv(nd):
    lo = px['low'].rolling(nd).min(); hi = px['high'].rolling(nd).max()
    return -(px['close'] - lo) / (hi - lo)

def sig_rev(nd):
    return -(np.log(px['close']) - np.log(px['close'].shift(nd)))

def sig_id_rev():
    return -(px['close'] / px['open'] - 1.0)

def sig_nbody():
    return -(px['close'] - px['open']) / (px['high'] - px['low'])

def sig_rev_vs(nd=1, vw=20):
    r = np.log(px['close']).diff()
    return -(np.log(px['close']) - np.log(px['close'].shift(nd))) / r.rolling(vw).std()

def sig_mom120():
    return px['close'].shift(5) / px['close'].shift(125) - 1.0

def sig_vix_beta(bw=60, vw=20):
    ret = np.log(px['close']).diff()
    vret = np.log(vix).diff()
    # align
    r = ret.copy(); vr = vret.reindex(ret.index)
    # rolling beta via cov/var
    cov = r.rolling(bw).cov(vr)
    var = vr.rolling(bw).var()
    beta = cov / var
    vix_move = vix / vix.shift(vw) - 1.0
    vix_move = vix_move.reindex(ret.index)
    return -beta * vix_move

def sig_volofvol(sw=20, lw=60):
    r = px['close'].pct_change()
    return r.rolling(sw).std().rolling(lw).std()

signals = {
    'miner2_20260715_nclv_1d': sig_nclv(1),
    'miner2_20260715_nclv_2d': sig_nclv(2),
    'miner2_20260715_nclv_3d': sig_nclv(3),
    'miner2_20260715_nclv_5d': sig_nclv(5),
    'miner2_20260715_rev_1d': sig_rev(1),
    'miner2_20260715_rev_2d': sig_rev(2),
    'miner2_20260715_rev_3d': sig_rev(3),
    'miner2_20260715_rev_5d': sig_rev(5),
    'miner2_20260715_rev_1d_vs': sig_rev_vs(),
    'miner2_20260715_id_rev_1d': sig_id_rev(),
    'miner2_20260715_nbody_1d': sig_nbody(),
    'mom_120d_skip5': sig_mom120(),
    'vix_beta_cond_60x20': sig_vix_beta(),
    'vol_of_vol20x60': sig_volofvol(),
}

# ---------- rank IC ----------
fwd = {}
for h in (1, 5, 10):
    fwd[h] = px['close'].shift(-h) / px['close'] - 1.0

def ic_stats(sig, h, n=60, ex_flat=True):
    f = fwd[h]
    valid = sig.notna() & f.notna()
    out = []
    for dt in valid.index:
        s = sig.loc[dt]; ff = f.loc[dt]
        m = s.notna() & ff.notna()
        if ex_flat:
            m = m & ~px.columns.isin(['HSI','CN10Y'])
        if m.sum() < 5:
            continue
        ic = s[m].corr(ff[m], method='spearman')
        if np.isfinite(ic):
            out.append((dt, ic))
    if len(out) < 20:
        return None
    ics = pd.Series([o[1] for o in out], index=[o[0] for o in out]).iloc[-n:]
    icm = ics.mean(); icsd = ics.std(ddof=1)
    icir = icm / icsd * np.sqrt(len(ics)) if icsd > 0 else 0.0
    hit = (ics > 0).mean()
    return {'ic_mean': round(float(icm), 4), 'icir': round(float(icir), 3),
            'hit': round(float(hit), 3), 'n': int(len(ics))}

res = {}
for name, sig in signals.items():
    res[name] = {}
    for h in (1, 5, 10):
        st = ic_stats(sig, h, n=60)
        res[name][f'ic{h}'] = st if st else {'ic_mean': None, 'icir': None, 'hit': None, 'n': 0}

json.dump(res, open('_screener_ic_20350615.json', 'w'), indent=1)
print()
for name in res:
    r = res[name]
    i1 = r['ic1']; i5 = r['ic5']; i10 = r['ic10']
    def fmt(x):
        if x['ic_mean'] is None: return '  NA  '
        return f"{x['ic_mean']:+.4f}/{x['icir']:+.2f}/h{x['hit']:.2f}"
    print(f"{name:34s} ic1 {fmt(i1)} | ic5 {fmt(i5)} | ic10 {fmt(i10)}")

# ---------- regime metrics ----------
r = np.log(px['close']).diff().dropna()
eqw20 = r.mean(axis=1).tail(20).mean() * 100
eqw20cum = (np.exp(r.mean(axis=1).tail(20).sum()) - 1) * 100
eqw60cum = (np.exp(r.mean(axis=1).tail(60).sum()) - 1) * 100
eqw120cum = (np.exp(r.mean(axis=1).tail(120).sum()) - 1) * 100
ma20 = px['close'].rolling(20).mean(); ma60 = px['close'].rolling(60).mean()
last = px['close'].iloc[-1]
br20 = (last > ma20.iloc[-1]).sum(); br60 = (last > ma60.iloc[-1]).sum()
vol20 = r.tail(20).std().mean() * np.sqrt(252) * 100
disp20 = r.tail(20).std(axis=1).mean() * 100
disp60 = r.tail(60).std(axis=1).mean() * 100
r20 = (last / px['close'].iloc[-21] - 1) * 100
leaders = r20.sort_values(ascending=False)
print()
print(f"eqw20d mean daily {eqw20:+.3f}% cum {eqw20cum:+.2f}% | 60d {eqw60cum:+.2f}% | 120d {eqw120cum:+.2f}%")
print(f"breadth MA20 {br20}/15 MA60 {br60}/15 | mean 20d ann vol {vol20:.1f}% | 20d x-sect disp {disp20:.2f}% (60d {disp60:.2f}%)")
print(f"VIX last {vix.iloc[-1]:.1f} (20d ago {vix.iloc[-21]:.1f}, 60d ago {vix.iloc[-61]:.1f})")
print("20d returns:"); print(leaders.round(2).to_string())
