"""Recompute the 3 active factors through 2032-06-25 and measure recent block IC."""
import pandas as pd, numpy as np

VD = '2032-06-25'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VD].sort_values('date').reset_index(drop=True)
    return df.set_index('date')['close']

px = pd.DataFrame({s: load(s) for s in ASSETS}).ffill()
rets = px.pct_change()

# --- factor 1: vol_adj_mom_accel_20x60 ---
vol20 = rets.rolling(20).std()
mom20 = px / px.shift(20) - 1
mom60 = px / px.shift(60) - 1
f1 = (mom20 - mom60) / vol20

# --- factor 2: dn_mkt_beta_60d ---
mkt = rets.mean(axis=1)
dn = mkt.where(mkt < 0, 0.0)

def roll_beta_df(y_df, x, w=60, min_obs=40):
    cols = y_df.columns
    out = pd.DataFrame(np.nan, index=y_df.index, columns=cols)
    arr_y = y_df.values; arr_x = x.values
    n = len(y_df)
    for i in range(w, n):
        seg_x = arr_x[i-w:i]
        base = ~np.isnan(seg_x)
        if base.sum() < min_obs:
            continue
        for j, c in enumerate(cols):
            seg_y = arr_y[i-w:i, j]
            mm = base & ~np.isnan(seg_y)
            if mm.sum() < min_obs:
                continue
            a, b = np.polyfit(seg_x[mm], seg_y[mm], 1)
            out.iloc[i, j] = b
    return out

f2 = roll_beta_df(rets, dn)

# --- factor 3: rate_beta_cn10y_60d ---
cn10y = px['CN10Y'].pct_change()
f3 = roll_beta_df(rets, cn10y)

# forward 10d returns
fwd10 = px.shift(-10) / px - 1

def rank_ic(f, y, min_valid=8):
    out = {}
    for dt in f.index:
        xv = f.loc[dt]; yv = y.loc[dt]
        m = xv.notna() & yv.notna()
        if m.sum() >= min_valid:
            out[dt] = np.corrcoef(xv[m].rank(), yv[m].rank())[0,1]
    return pd.Series(out)

ic1 = rank_ic(f1, fwd10); ic2 = rank_ic(f2, fwd10); ic3 = rank_ic(f3, fwd10)

print('=== RECENT FACTOR IC (rank IC vs fwd 10d, universe>=8 valid) ===')
for name, ic in [('vol_adj_mom_accel_20x60', ic1), ('dn_mkt_beta_60d', ic2), ('rate_beta_cn10y_60d', ic3)]:
    recent = ic[ic.index >= '2031-12-01']
    last3 = ic[ic.index >= '2032-04-01']
    print('%s | n=%d | 2020+ mean IC %.3f ICIR %.2f hit %.2f | since 2031-12 mean %.3f hit %.2f | since 2032-04 mean %.3f hit %.2f' % (
        name, ic.notna().sum(), ic.mean(), ic.mean()/ic.std() if ic.std()>0 else 0, (ic>0).mean(),
        recent.mean(), (recent>0).mean(), last3.mean(), (last3>0).mean()))

print('\n=== LATEST FACTOR EXPOSURES (2032-06-25) ===')
for name, f in [('vol_adj_mom_accel_20x60', f1), ('dn_mkt_beta_60d', f2), ('rate_beta_cn10y_60d', f3)]:
    row = f.iloc[-1]
    print(name)
    print('  top5:', [(a, round(row[a],4)) for a in row.sort_values(ascending=False).index[:5]])
    print('  bot5:', [(a, round(row[a],4)) for a in row.sort_values(ascending=False).index[-5:]])

print('\n=== LATEST EXPOSURE PAIRWISE CORR ===')
stack = pd.concat([f1.iloc[-1].rename('f1'), f2.iloc[-1].rename('f2'), f3.iloc[-1].rename('f3')], axis=1)
print(stack.corr().round(3))
