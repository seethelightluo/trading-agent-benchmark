"""Shared harness for factor validation - miner_2. Use data visible through 2035-11-06."""
import os, numpy as np, pandas as pd

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA = "../persistent/stock_data"
END = "2035-11-06"
START = "2020-01-01"

def load_panels(end=END, start=START, include_macro=True):
    closes = {}
    for s in WATCH:
        df = pd.read_csv(os.path.join(DATA, s+".csv"), parse_dates=["date"]).set_index("date")
        df = df[~df.index.duplicated(keep="last")]
        closes[s] = df["close"]
    px = pd.DataFrame(closes).sort_index()
    px = px[(px.index>=pd.Timestamp(start)) & (px.index<=pd.Timestamp(end))]
    ret = px.pct_change()
    obs = None
    if include_macro:
        obs = {}
        for s in ["DXY","USDCNY","USDJPY","EURUSD","VIX"]:
            df = pd.read_csv(os.path.join("../persistent/index_data", s+".csv"),
                             parse_dates=["date"]).set_index("date")
            df = df[~df.index.duplicated(keep="last")]
            obs[s] = df["close"]
        obs = pd.DataFrame(obs).sort_index()
        obs = obs[(obs.index>=pd.Timestamp(start)) & (obs.index<=pd.Timestamp(end))]
    return px, ret, obs

def cross_sectional_ic(signal, forward_ret, min_assets=8):
    sig = signal.reindex(index=forward_ret.index, columns=forward_ret.columns)
    fwd = forward_ret.reindex(index=sig.index, columns=sig.columns)
    dates, ics = [], []
    for dt in fwd.index:
        x, y = sig.loc[dt], fwd.loc[dt]
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < min_assets: continue
        if x[m].std() < 1e-9 or y[m].std() < 1e-9: continue
        ic = np.corrcoef(x[m], y[m])[0,1]
        if np.isfinite(ic): dates.append(dt); ics.append(ic)
    return pd.Series(ics, index=pd.Index(dates, name="date"))

def compute_forward_rets(rets, horizon):
    return (1+rets).rolling(horizon).apply(lambda x: x.prod()-1, raw=True).shift(-horizon)

def report(name, fac, ret, IDX, horizon=10):
    fwd = compute_forward_rets(ret, horizon)
    s = cross_sectional_ic(fac, fwd)
    if len(s)==0: return None
    mu = float(s.mean()); sd = float(s.std())
    icir = mu/sd if sd>0 else 0.0
    hit = float((s>0).mean())
    rec = s[s.index>="2034-11-01"]
    rmu = float(rec.mean()) if len(rec) else np.nan
    rsd = float(rec.std()) if len(rec)>2 else np.nan
    ricir = rmu/rsd if rsd and rsd>0 else np.nan
    dec = []
    for h in [1,2,3,5,10,20]:
        f = compute_forward_rets(ret, h)
        ss = cross_sectional_ic(fac, f)
        dec.append(round(float(ss.mean()),4) if len(ss) else np.nan)
    gate = "PASS" if (abs(mu)>=0.0070 and abs(icir)>=0.0840) else "fail"
    # turnover proxy: mean abs daily change of cross-sectional rank
    fr = fac.reindex(IDX)
    rank = fr.rank(axis=1)
    turn = float(rank.diff().abs().mean(axis=1).mean()) if len(IDX)>2 else np.nan
    cov = float(fr.notna().mean().mean())
    print(f"{name}: n={len(s)} IC={mu:.4f} ICIR={icir:.4f} hit={hit:.3f} | 1y IC={rmu:.4f} ICIR={ricir:.4f} | decay={dec} | turn={turn:.3f} cov={cov:.3f} | full[{gate}]")
    return dict(ic=mu, icir=icir, hit=hit, n=len(s), rec_ic=rmu, rec_icir=ricir,
                decay=dec, turnover_rank_proxy=turn, coverage=cov, gate=gate)

if __name__ == "__main__":
    px, ret, obs = load_panels()
    IDX = ret.index
    print("rows", len(px), "last", px.index.max())
    vix = obs["VIX"]; vixr = vix.pct_change()
    print("VIX last %.1f 60d avg %.1f" % (vix.iloc[-1], vix[-60:].mean()))
    print("SPX 20d % .2f" % ((px['SPX'].iloc[-1]/px['SPX'].iloc[-21]-1)*100))