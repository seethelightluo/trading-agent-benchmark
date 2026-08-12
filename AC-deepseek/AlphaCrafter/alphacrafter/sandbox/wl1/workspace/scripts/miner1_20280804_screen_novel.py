"""miner_1 2028-08-04: screen novel factor ideas (vectorized IC).
Validation window: 2020-01-01 .. 2028-08-03. Gate: |IC10|>=0.0070 & |ICIR10|>=0.0840."""
import pandas as pd, numpy as np, pickle, json, base64, zlib, io

with open('scripts/_miner1_panel_20280804.pkl','rb') as fh:
    P = pickle.load(fh)
close = P['close']; volume = P['volume']
WATCH = list(close.columns)
ret = close.pct_change()

def load_macro(name):
    m = pd.read_csv(f'../persistent/index_data/{name}.csv')
    m['date'] = pd.to_datetime(m['date']); m = m.set_index('date').sort_index()
    return m['close'].astype(float)
spx = close['SPX']; us10y = close['US10Y']

def ma(s, w): return s.rolling(w).mean()
def std(s, w): return s.rolling(w).std()

F = {}
F['risk_adj_mom_120_20'] = (close.shift(5)/close.shift(125)-1.0) / std(ret,20)
F['trend_slope_20_60'] = (ma(close,20)/ma(close,60)-1.0)
F['ma20_dist_z'] = (close/ma(close,20)-1.0) / std(ret,20)
F['dd_60d'] = close/close.rolling(60).max()-1.0
F['range_pos_20d'] = (close-close.rolling(20).min())/(close.rolling(20).max()-close.rolling(20).min())
F['neg_beta_spx_60'] = -ret.rolling(60).cov(spx.pct_change())/spx.pct_change().rolling(60).var()
dr = ret.where(ret<0, 0.0)
F['neg_downside_vol_20'] = -dr.rolling(20).std()
F['vol_adj_rev_1d'] = -(ret)/std(ret,20)
F['mom_trend_interact'] = (close.shift(5)/close.shift(125)-1.0) * (close>ma(close,60)).astype(float)
F['streak_5d'] = (ret>0).astype(float).rolling(5).sum()-(ret<0).astype(float).rolling(5).sum()
d10 = us10y.pct_change()
F['neg_rate_beta_60'] = -ret.rolling(60).cov(d10)/d10.rolling(60).var()
rv20 = std(ret,20)
F['neg_vol_z_20'] = -(rv20-rv20.rolling(120).mean())/rv20.rolling(120).std()
volr = volume.rolling(20).mean()/volume.rolling(60).mean()
F['volconf_mom_20'] = (close/close.shift(20)-1.0) * volr

H = [1,2,3,5,10,20]
fwd_ranks = {}
for h in H:
    fr = close.shift(-h)/close - 1.0
    fwd_ranks[h] = fr.rank(axis=1)

def fast_ic(fv, h):
    """Vectorized daily spearman IC between factor values and h-day fwd return."""
    Rf = fv.rank(axis=1)
    Ry = fwd_ranks[h]
    valid = fv.notna() & (close.shift(-h).notna())
    nv = valid.sum(axis=1)
    ok = nv >= 8
    ic = pd.Series(np.nan, index=fv.index)
    if not ok.any():
        return ic
    # centered products
    rf = Rf[ok].values; ry = Ry[ok].values; v = valid[ok].values
    rf = np.where(v, rf, np.nan); ry = np.where(v, ry, np.nan)
    mf = np.nanmean(rf, axis=1, keepdims=True); my = np.nanmean(ry, axis=1, keepdims=True)
    cf = np.where(v, rf-mf, np.nan); cy = np.where(v, ry-my, np.nan)
    num = np.nansum(cf*cy, axis=1)
    den = np.sqrt(np.nansum(cf**2, axis=1)*np.nansum(cy**2, axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        icv = num/den
    ic.loc[ok] = icv
    return ic

def metrics(fid, fv):
    res = {}
    ics = {}
    for h in H:
        ic = fast_ic(fv, h)
        ics[h] = ic
        res[f'ic{h}'] = ic.mean() if len(ic) else np.nan
        res[f'icir{h}'] = ic.mean()/ic.std() if ic.std()>0 else np.nan
        res[f'n{h}'] = int(ic.notna().sum())
    ic10 = ics[10]
    res['hit10'] = (ic10>0).mean()
    valid = fv.notna()
    res['coverage_asset_days'] = valid.sum().sum()/(valid.shape[0]*valid.shape[1])
    res['coverage_dates_ge8'] = (valid.sum(axis=1)>=8).mean()
    rk = fv.rank(axis=1); rk10 = rk.iloc[::10]
    res['turnover_10d_rank'] = rk10.diff().abs().mean().mean()
    cut = pd.Timestamp('2026-07-16')
    for nm, mask in (('warmup', ic10.index < cut), ('online', ic10.index >= cut)):
        sub = ic10[mask].dropna()
        res[f'ic10_{nm}'] = sub.mean() if len(sub) else np.nan
        res[f'icir10_{nm}'] = sub.mean()/sub.std() if sub.std()>0 else np.nan
        res[f'n10_{nm}'] = int(len(sub))
    return res

def load_lib_signal(fid):
    d = json.load(open(f'factors/{fid}.json'))
    a = d.get('validation',{}).get('signal_artifact',{})
    if not a or not a.get('data'): return None
    fmt = a.get('format','')
    if 'zlib' not in fmt: return None
    txt = zlib.decompress(base64.b64decode(a['data'])).decode()
    return pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)

lib_ids = ['mom_120d_skip5','vol_of_vol20x60','vix_beta_cond_60x20']
lib_sigs = {fid: load_lib_signal(fid) for fid in lib_ids}
lib_sigs = {k:v for k,v in lib_sigs.items() if v is not None}
print('loaded lib signals:', list(lib_sigs.keys()))

def lib_corr(fv):
    best = {}
    for fid, ldf in lib_sigs.items():
        ci = fv.index.intersection(ldf.index)
        cc = [c for c in WATCH if c in ldf.columns]
        a = fv.loc[ci, cc].values.ravel(); b = ldf.loc[ci, cc].values.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() > 100:
            best[fid] = float(np.corrcoef(a[m], b[m])[0,1])
    return best, (max(abs(v) for v in best.values()) if best else None)

rows = []
for fid, fv in F.items():
    m = metrics(fid, fv)
    corr, mx = lib_corr(fv)
    m['max_abs_library_correlation'] = mx
    m['lib_corr'] = corr
    rows.append((fid, m))

print('\n==== SCREEN RESULTS (2020-01-01..2028-08-03, h=10 admission) ====')
hdr = f'{"factor":24s} {"ic10":>7s} {"icir10":>7s} {"hit10":>5s} {"n10":>5s} {"covAD":>5s} {"turn":>5s} {"ic1":>7s} {"ic20":>7s} {"ic10_warm":>9s} {"ic10_onl":>9s} {"maxRho":>6s}'
print(hdr)
for fid, m in rows:
    print(f'{fid:24s} {m["ic10"]:7.4f} {m["icir10"]:7.3f} {m["hit10"]:5.2f} {m["n10"]:5d} {m["coverage_asset_days"]:5.2f} {m["turnover_10d_rank"]:5.2f} {m["ic1"]:7.4f} {m["ic20"]:7.4f} {m["ic10_warmup"]:9.4f} {m["ic10_online"]:9.4f} {m["max_abs_library_correlation"]:6.3f}')

with open('scripts/_miner1_screen_20280804.json','w') as fh:
    json.dump({fid: {k:v for k,v in m.items() if k!='lib_corr'} for fid,m in rows}, fh, indent=1, default=str)
print('saved')
