"""miner_1 2035-05-18: debug DXY-beta conditional factor (previous screen showed n=0 dates)."""
import numpy as np
import pandas as pd

panel = pd.read_pickle('scripts/panel_cache_20350518.pkl')
px = panel['close']; ret = panel['ret']
mac = panel['macro']

dxy = mac['DXY'].reindex(px.index).ffill()
dxy_ret = dxy.pct_change()
print("dxy_ret valid count:", int(dxy_ret.notna().sum()))
print("ret shape:", ret.shape)

# ---- replicate library-style computation ----
betas = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for i in range(60, len(ret)):
    a = ret.iloc[i-60:i]          # 60 x 15
    b = dxy_ret.iloc[i-60:i]      # 60 (Series)
    m = a.notna() & b.notna()
    if int(m.sum().sum()) < 10:
        continue
    aa = a[m]
    bb = b[m]
    print(f"i={i} date={px.index[i].date()} m.sum={int(m.sum().sum())} aa.shape={aa.shape} bb.shape={bb.shape} bb type={type(bb)}")
    if i > 100:
        break

# ---- alternative: compute per-column beta with aligned series ----
print("\n--- alternative per-column beta (aligned) ---")
betas2 = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for i in range(60, len(ret)):
    for c in ret.columns:
        a = ret[c].iloc[i-60:i]
        b = dxy_ret.iloc[i-60:i]
        m = a.notna() & b.notna()
        if int(m.sum()) < 10:
            continue
        aa = a[m]; bb = b[m]
        cov = float(np.cov(aa, bb)[0, 1])
        var = float(np.var(bb))
        if var > 0:
            betas2.iloc[i, betas2.columns.get_loc(c)] = cov / var

dxy_trend = dxy_ret.rolling(20).mean()
cond2 = betas2 * np.sign(dxy_trend).values[:, None]
print("betas2 non-NaN count:", int(betas2.notna().sum().sum()))
print("cond2 non-NaN count:", int(cond2.notna().sum().sum()))
print("dates with >=8 valid cond2:", int(cond2.notna().sum(axis=1).ge(8).sum()))
print("cond2 last date sample (2035-05-17):")
print(cond2.loc[px.index[-1]].dropna().round(4).to_dict())
print("\ncond2 last 5 rows sample (first 5 assets):")
print(cond2.tail(5).iloc[:, :5].round(4))
