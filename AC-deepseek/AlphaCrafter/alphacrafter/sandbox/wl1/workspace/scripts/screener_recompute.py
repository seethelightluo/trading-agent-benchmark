import pandas as pd, numpy as np, json

base = '../persistent/stock_data/'
assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
PX = {}
for a in assets:
    df = pd.read_csv(base+a+'.csv')
    df.columns=[c.strip() for c in df.columns]
    df['date']=pd.to_datetime(df['date'])
    df=df.sort_values('date').set_index('date')
    PX[a]=df
C = pd.DataFrame({a: PX[a]['close'] for a in assets}).sort_index()
O = pd.DataFrame({a: PX[a]['open'] for a in assets}).reindex(C.index)
H = pd.DataFrame({a: PX[a]['high'] for a in assets}).reindex(C.index)
L = pd.DataFrame({a: PX[a]['low'] for a in assets}).reindex(C.index)
C = C[C.index <= '2035-12-13']
O = O.reindex(C.index); H = H.reindex(C.index); L = L.reindex(C.index)

vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix.columns=[c.strip() for c in vix.columns]
vix['date']=pd.to_datetime(vix['date'])
vix=vix.sort_values('date').set_index('date')['close']
vix=vix[vix.index <= '2035-12-13']

rets = C.pct_change()
lnc = np.log(C)

def spearman(x, y):
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/den) if den>0 else 0.0

def rank_ic(factor, fwd, min_cov=8):
    fwd_ret = C.shift(-fwd)/C - 1.0
    common = factor.index.intersection(fwd_ret.index)
    ics = []
    for dt in common:
        f = factor.loc[dt]; r = fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() < min_cov: continue
        ics.append(spearman(f[m], r[m]))
    return np.array(ics)

F = {}
F['miner2_20260715_id_rev_1d'] = -(C/O - 1.0)
F['miner2_20260715_nbody_1d'] = -(C-O)/(H-L).replace(0,np.nan)
for nd in [1,2,3,5]:
    lo = L.rolling(nd).min(); hi = H.rolling(nd).max()
    F[f'miner2_20260715_nclv_{nd}d'] = -(C-lo)/(hi-lo).replace(0,np.nan)
for nd in [1,2,3,5]:
    F[f'miner2_20260715_rev_{nd}d'] = -(lnc - lnc.shift(nd))
F['miner2_20260715_rev_1d_vs'] = -(lnc - lnc.shift(1))/rets.rolling(20).std().replace(0,np.nan)
F['mom_120d_skip5'] = C.shift(5)/C.shift(125) - 1.0
vixr = vix.pct_change()
vix_ratio = vix/vix.shift(20) - 1.0
vr = vixr.reindex(C.index)
vratio = vix_ratio.reindex(C.index)
beta60 = rets.rolling(60).cov(vr).div(vr.rolling(60).var().replace(0,np.nan), axis=0)
F['vix_beta_cond_60x20'] = -beta60 * vratio
F['vol_of_vol20x60'] = rets.rolling(20).std().rolling(60).std()

out = {}
for fid, panel in F.items():
    panel = panel.reindex(C.index)
    rec = {}
    for h in [1,5,10]:
        ics = rank_ic(panel, h)
        if len(ics) < 30:
            rec[f'ic{h}'] = {'ic_mean': None, 'icir': None, 'hit': None, 'n': int(len(ics))}
            continue
        last60 = ics[-60:]; last120 = ics[-120:]
        rec[f'ic{h}'] = {
            'ic_mean': round(float(last60.mean()),4),
            'icir': round(float(last60.mean()/last60.std()),3) if last60.std()>0 else 0.0,
            'hit': round(float((last60>0).mean()),3),
            'n': int(len(last60)),
            'ic_mean_120': round(float(last120.mean()),4),
            'n120': int(len(last120)),
            'ic_last': round(float(ics[-1]),4),
        }
    out[fid] = rec

json.dump(out, open('_screener_ic_fresh.json','w'), indent=1)
print("wrote _screener_ic_fresh.json")
for fid, rec in out.items():
    if rec.get('ic10',{}).get('ic_mean') is not None:
        print(f"{fid:35s} ic10 {rec['ic10']['ic_mean']:+.4f}/{rec['ic10']['icir']:+.2f} hit{rec['ic10']['hit']:.2f} n{rec['ic10']['n']} | ic5 {rec['ic5']['ic_mean']:+.4f}/{rec['ic5']['icir']:+.2f} | ic1 {rec['ic1']['ic_mean']:+.4f}/{rec['ic1']['icir']:+.2f}")
    else:
        print(f"{fid:35s} ic10 n<30 -> {rec.get('ic10',{}).get('n')}")
