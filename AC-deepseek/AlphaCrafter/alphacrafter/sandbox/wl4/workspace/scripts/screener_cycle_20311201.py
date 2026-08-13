"""SCREENER cycle 2031-12-01: regime assessment + fresh factor IC/ICIR on 3 active factors.
Uses ONLY data through visible_through=2031-11-28 (no future data)."""
import pandas as pd, numpy as np, json, os

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF = '2031-11-28'

def load(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= CUTOFF].reset_index(drop=True)
    df = df.set_index('date').sort_index()
    return df['close']

px = pd.DataFrame({s: load(s) for s in ASSETS})
px = px.dropna(how='all')
print('price panel rows:', len(px), 'cols:', len(px.columns), 'range:', px.index.min().date(), '->', px.index.max().date())

ret = px.pct_change()

# ---------- regime metrics ----------
last = px.iloc[-1]
ma20 = px.rolling(20).mean().iloc[-1]
ma60 = px.rolling(60).mean().iloc[-1]
r20 = px.iloc[-1] / px.iloc[-21] - 1
r60 = px.iloc[-1] / px.iloc[-61] - 1

print('\n=== ASSET STATE @2031-11-28 ===')
for s in ASSETS:
    flat = (px[s].iloc[-40:].nunique() <= 2)
    print(f'{s:10s} close={last[s]:12.3f} 20d={r20[s]*100:+7.2f}% 60d={r60[s]*100:+7.2f}% '
          f'vsMA20={100*(last[s]/ma20[s]-1):+6.2f}% vsMA60={100*(last[s]/ma60[s]-1):+6.2f}% flat40d={flat}')

# realized vol (20d, annualized) cross-section
vol20 = ret.tail(20).std() * np.sqrt(252)
print('\n20d realized vol (ann):')
for s in ASSETS:
    print(f'  {s:10s} {vol20[s]*100:6.1f}%')

# VIX
vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix['date'] = pd.to_datetime(vix['date'])
vix = vix[vix['date'] <= CUTOFF].set_index('date').sort_index()
print('\nVIX last:', round(vix['close'].iloc[-1],1), '20d mean:', round(vix['close'].tail(20).mean(),1),
      '60d mean:', round(vix['close'].tail(60).mean(),1), 'max60:', round(vix['close'].tail(60).max(),1))

# cross-sectional dispersion (20d ret cross-section std) and mean pairwise corr (60d)
csd = ret.tail(20).std(axis=1)
print('\nCross-sectional dispersion (daily std of 20d asset rets): last20 mean =', round(csd.mean()*100,3), 'last =', round(csd.iloc[-1]*100,3))
c = ret.tail(60).corr()
vals = c.values[np.triu_indices(len(c),1)]
print('Mean pairwise 60d corr:', round(np.nanmean(vals),3), 'max:', round(np.nanmax(vals),3))

# breadth
above_ma60 = sum(last[s] > ma60[s] for s in ASSETS)
print('Breadth above MA60:', above_ma60, '/', len(ASSETS))

# ---------- factor signals ----------
def zscore(s):
    return (s - s.mean()) / s.std()

# 1) vol_adj_mom_accel_20x60
mom20 = px / px.shift(20) - 1
mom60 = px / px.shift(60) - 1
volr = ret.rolling(20).std()
sig1 = (mom20 - mom60) / volr

# 2) dn_mkt_beta_60d
mkt = ret.mean(axis=1)
dn = mkt.where(mkt < 0, 0.0)
cov = ret.rolling(60).cov(dn)
var = dn.rolling(60).var()
sig2 = cov / var

# 3) rate_beta_cn10y_60d
cn = px['CN10Y']
dcn = cn.pct_change()
cov3 = ret.rolling(60).cov(dcn)
var3 = dcn.rolling(60).var()
sig3 = cov3 / var3

sigs = {'vol_adj_mom_accel_20x60': sig1, 'dn_mkt_beta_60d': sig2, 'rate_beta_cn10y_60d': sig3}

# ---------- IC / ICIR at h=10 over recent 250d window ----------
H = 10
fwd = px.shift(-H) / px - 1
W = 250
print('\n=== FACTOR METRICS (recent %dd window, h=%d) ===' % (W, H))
out = {}
for fid, sig in sigs.items():
    aligned = pd.concat([sig.stack(), fwd.stack()], axis=1, keys=['sig','fwd']).dropna()
    aligned = aligned[aligned.index.get_level_values(0) >= aligned.index.get_level_values(0).max() - pd.Timedelta(days=400)]
    # restrict to last W unique trading days
    days = sorted(aligned.index.get_level_values(0).unique())[-W:]
    a = aligned[aligned.index.get_level_values(0).isin(days)]
    ic_by_day = a.groupby(level=0).apply(lambda g: g['sig'].corr(g['fwd'], method='spearman'))
    ic = ic_by_day.mean()
    icir = ic_by_day.mean() / ic_by_day.std() * np.sqrt(len(ic_by_day)/ (252/10)) if ic_by_day.std() > 0 else np.nan
    # simpler: ICIR = mean/std of daily ICs (annualized by sqrt(252/10))
    icir2 = ic_by_day.mean() / ic_by_day.std() * np.sqrt(252/H) if ic_by_day.std() > 0 else np.nan
    q = abs(ic) * abs(icir2)
    out[fid] = dict(ic=ic, icir=icir2, q=q, n=len(ic_by_day), last_sig=float(sig.iloc[-1].mean()))
    print(f'{fid:24s} IC={ic:+.4f} ICIR={icir2:+.3f} q={q:.4f} ndays={len(ic_by_day)}')

# factor pairwise correlation (recent cross-section, latest date)
print('\n=== FACTOR PAIRWISE CORR (latest cross-section) ===')
last_date = px.index[-1]
fids = list(sigs.keys())
for i in range(len(fids)):
    for j in range(i+1, len(fids)):
        a = sigs[fids[i]].loc[last_date]
        b = sigs[fids[j]].loc[last_date]
        c = np.corrcoef(a, b)[0,1]
        print(f'  {fids[i]:24s} vs {fids[j]:24s}: {c:+.3f}')

# turnover proxy: mean abs change in cross-sectional rank per 10d
print('\n=== TURNOVER (mean |rank change| per 10d, recent 120d) ===')
for fid, sig in sigs.items():
    rk = sig.rank(axis=1)
    chg = rk.diff(10).abs().mean().mean()
    print(f'  {fid:24s} mean|dRank|/10d = {chg:.2f}')

# stale names check: last 40d unique closes
print('\n=== STALE CHECK (unique closes in last 40d) ===')
for s in ASSETS:
    u = px[s].tail(40).nunique()
    print(f'  {s:10s} unique={u}')
