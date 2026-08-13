import pandas as pd, numpy as np

ASOF = "2034-03-20"
assets = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load(p):
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df

px = {}
for a in assets:
    df = load(f"../persistent/stock_data/{a}.csv")
    df = df[df.index <= ASOF]
    px[a] = df["close"].astype(float)
PX = pd.DataFrame(px)
RET = PX.pct_change()

frozen = [a for a in assets if PX[a].tail(60).nunique() <= 2]
liquid = [a for a in assets if a not in frozen]
print("frozen:", frozen, "| liquid:", liquid)

# ---- factor computations (on full 15 then mask) ----
# 1) vol_adj_mom_accel_20x60 = (mom20 - mom60)/vol20
mom20 = PX / PX.shift(20) - 1
mom60 = PX / PX.shift(60) - 1
vol20 = RET.rolling(20).std()
f_mom = (mom20 - mom60) / vol20

# 2) dn_mkt_beta_60d: beta on down-market days
mkt = RET.mean(axis=1)
down = mkt.where(mkt < 0)
f_beta = pd.DataFrame(index=RET.index, columns=PX.columns, dtype=float)
for a in assets:
    x = down; y = RET[a]
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 40: continue
    # rolling 60d beta of y on x (down days only)
    betas = []
    idx = df.index
    for i in range(len(df)):
        w = df.iloc[max(0,i-59):i+1]
        if len(w) >= 40 and w.iloc[:,0].std() > 0:
            b = np.cov(w.iloc[:,0], w.iloc[:,1])[0,1] / w.iloc[:,0].var()
            betas.append((idx[i], b))
        else:
            betas.append((idx[i], np.nan))
    f_beta[a] = pd.Series(dict(betas)).reindex(RET.index)

# 3) rate_beta_cn10y_60d: beta on CN10Y pct change
cn = PX["CN10Y"].pct_change()
f_rate = pd.DataFrame(index=RET.index, columns=PX.columns, dtype=float)
for a in assets:
    x = cn; y = RET[a]
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 40: continue
    betas = []
    idx = df.index
    for i in range(len(df)):
        w = df.iloc[max(0,i-59):i+1]
        if len(w) >= 40 and w.iloc[:,0].std() > 0:
            b = np.cov(w.iloc[:,0], w.iloc[:,1])[0,1] / w.iloc[:,0].var()
            betas.append((idx[i], b))
        else:
            betas.append((idx[i], np.nan))
    f_rate[a] = pd.Series(dict(betas)).reindex(RET.index)

# ---- rolling 10d-forward rank IC over last 120 trading days ----
fwd = RET.shift(-10)  # 10d forward return (t -> t+10)
fwd10 = (PX.shift(-10)/PX - 1)

def ic_series(factor, fwd10, n_min=8):
    out = {}
    for d in factor.index:
        x = factor.loc[d]; y = fwd10.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= n_min:
            out[d] = np.corrcoef(x[m].rank(), y[m].rank())[0,1]
    return pd.Series(out)

print("\n=== Rolling 10d-forward rank IC, last 120 trading days ===")
for name, f in [("vol_adj_mom_accel_20x60", f_mom), ("dn_mkt_beta_60d", f_beta), ("rate_beta_cn10y_60d", f_rate)]:
    s = ic_series(f, fwd10)
    s = s[s.index >= s.index[-1] - pd.Timedelta(days=200)]
    print(f"{name:>24}: n={len(s):3d}  mean_ic={s.mean():+.4f}  icir={s.mean()/s.std():+.3f}  hit={ (s>0).mean():.2f}  last10d_ic={s.tail(10).mean():+.4f}")

# also current factor values
print("\n=== Current factor values (2034-03-20) ===")
for name, f, sign in [("mom_accel", f_mom, 1), ("dn_beta", f_beta, 1), ("rate_beta", f_rate, -1)]:
    print(f"-- {name} (dir {sign}) --")
    row = f.loc[ASOF].dropna().sort_values()
    print(row.round(3).to_string())

# recent 10d returns for comparison
print("\n-- 10d returns (2034-03-20) --")
print((fwd10.loc[ASOF]*100).sort_values().round(2).to_string())
