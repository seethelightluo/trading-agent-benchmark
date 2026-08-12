import pandas as pd, numpy as np
from scipy.stats import spearmanr

ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
VIS = '2029-11-02'
HORIZON = 10

def load(sym, col='close'):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= VIS].set_index('date')[col]
    return df

closes = {s: load(s) for s in ASSETS}
px = pd.DataFrame(closes).dropna(how='all')
ret = px.pct_change()
mkt = ret.mean(axis=1)  # equal-weight 15-asset market

# ---- factor 1: dn_mkt_beta_60d ----
down = ret.where(mkt < 0)
cov = down.rolling(60, min_periods=40).cov(mkt)
var = mkt.where(mkt<0).rolling(60, min_periods=40).var()
f_dn = cov.div(var, axis=0)

# ---- factor 2: rate_beta_cn10y_60d ----
cn10y = px['CN10Y']
dcn = cn10y.pct_change()
cov2 = ret.rolling(60, min_periods=40).cov(dcn)
var2 = dcn.rolling(60, min_periods=40).var()
f_rate = cov2.div(var2, axis=0)

# ---- factor 3: vol_adj_mom_accel_20x60 ----
mom20 = px / px.shift(20) - 1
mom60 = px / px.shift(60) - 1
vol20 = ret.rolling(20).std()
f_mom = (mom20 - mom60) / vol20

factors = {'dn_mkt_beta_60d': f_dn, 'rate_beta_cn10y_60d': f_rate, 'vol_adj_mom_accel_20x60': f_mom}
dirs = {'dn_mkt_beta_60d': 1, 'rate_beta_cn10y_60d': -1, 'vol_adj_mom_accel_20x60': 1}

fwd = px.shift(-HORIZON) / px - 1

frozen = {'000300.SH','HSI','ETH'}
for name, sig in factors.items():
    ic_series = []
    # iterate over evaluation dates, last ~180 trading days
    dates = sig.index[60:-HORIZON]
    dates = dates[-180:]
    for t in dates:
        s = sig.loc[t].dropna()
        fr = fwd.loc[t].dropna()
        common = s.index.intersection(fr.index)
        common = [c for c in common if c not in frozen]
        if len(common) < 8:
            continue
        ic, _ = spearmanr(s[common], fr[common])
        if np.isfinite(ic):
            ic_series.append((t, ic))
    if not ic_series:
        print(name, 'no data'); continue
    ic_df = pd.Series([x[1] for x in ic_series], index=[x[0] for x in ic_series])
    last60 = ic_df[ic_df.index >= ic_df.index[-1] - pd.Timedelta(days=120)].tail(60)
    print(f'=== {name} (dir {dirs[name]:+d}) ===')
    print(f'  last 180d: mean IC {ic_df.mean():+.4f} | ICIR {ic_df.mean()/ic_df.std():+.3f} | n={len(ic_df)}')
    print(f'  last 60d : mean IC {last60.mean():+.4f} | ICIR {last60.mean()/last60.std():+.3f} | n={len(last60)}')
    print(f'  recent 10 ICs: {np.round(ic_df.tail(10).values, 3)}')
    q = abs(ic_df.mean()) * abs(ic_df.mean()/ic_df.std())
    print(f'  q=abs(IC)*abs(ICIR) last180d: {q:.4f}')
