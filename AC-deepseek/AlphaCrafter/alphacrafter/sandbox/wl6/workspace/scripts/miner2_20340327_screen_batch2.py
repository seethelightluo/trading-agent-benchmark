"""miner_2 screen batch 2 (2034-03-28), VIS 2034-03-27. Additional candidates.
Gate: |IC|>=0.0070 and |ICIR|>=0.0840 at horizon 10.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, TRADABLE
import pandas as pd, numpy as np, math

VIS = "2034-03-27"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill()
px = px.dropna(how="all").dropna(axis=1, how="all")
ret = px.pct_change()

def evalc(f, label):
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic)>1 else np.nan
    icir = icm/icstd if icstd and math.isfinite(icstd) and icstd>0 else np.nan
    hit = float((ic>0).mean())
    recent = ic[ic.index >= "2032-03-28"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent2y_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} GATE={'PASS' if gate else 'fail'}")
    assert isinstance(f, pd.DataFrame), label

cands = {}
spx = px["SPX"]

# H. 120d momentum dived by realized 20d vol (risk-scaled long trend) - smoother version
mom120 = px/px.shift(120)-1
vol20 = ret.rolling(20).std().replace(0,np.nan)
cands["mom120_div_vol20"] = mom120/vol20

# I. Idiosyncratic momentum: residual after removing SPX beta using 60d regression
beta_spx = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    beta_spx[a] = ret[a].rolling(60).cov(ret["SPX"]) / ret["SPX"].rolling(60).var()
resid = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for a in px.columns:
    resid[a] = ret[a] - beta_spx[a] * ret["SPX"]
cands["idio_mom20_skip5"] = resid.rolling(20).sum() - resid.shift(5).rolling(20).sum()

# J. Vol ratio 10x60 (mean-reversion when short vol spikes)
cands["vol_ratio_10x60"] = ret.rolling(10).std()/ret.rolling(60).std().replace(0,np.nan)

# K. 5d short-term reversal (negated: momentum after pullback)
cands["rev_5d_neg"] = -px.pct_change(5)

# L. Max return 20d trail (momentum persistence, downside-focused upward tail)
cands["max_ret20"] = pd.DataFrame({a: ret[a].rolling(20).max() for a in px.columns})

# M. Distance from 200d mean (long-term reversion)
m200 = px.rolling(200).mean()
cands["dist_ma200"] = (px - m200)/m200.replace(0,np.nan)

# N. XAU momentum minus WTI momentum (precious-industrial relative trend)
cands["xau_minus_wti20"] = px["XAU"].pct_change(20) - px["WTI"].pct_change(20)

# O. Equity minus CTA/haven: average equity index 20d mom minus XAU 20d mom
eqc = px[["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX"]]
cands["eq_minus_xau20"] = eqc.pct_change(20).mean(axis=1) - px["XAU"].pct_change(20)

# P. Crypto-commodity relative (risk appetite)
cryp = px[["BTC","ETH"]]
cands["crypto_minus_comm20"] = cryp.pct_change(20).mean(axis=1) - px[["XAU","COPPER","WTI"]].pct_change(20).mean(axis=1)

# Q. Cross-sectional winner minus loser (relative strength spread reversion)
mom20 = px.pct_change(20)
def rs_disp(x):
    r = x.rank(axis=1, pct=True)
    return (r - 0.5)
cands["rel_str_rank20"] = pd.DataFrame({a: mom20[a].rank(pct=True) for a in mom20.columns}, index=mom20.index)

for name, f in cands.items():
    evalc(f, name)