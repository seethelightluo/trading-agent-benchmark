import pandas as pd, numpy as np, math, json

VISIBLE = pd.Timestamp('2035-03-14')
WATCH = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
MACRO = {"VIX":"../persistent/index_data/VIX.csv","DXY":"../persistent/index_data/DXY.csv",
         "USDCNY":"../persistent/index_data/USDCNY.csv","USDJPY":"../persistent/index_data/USDJPY.csv",
         "EURUSD":"../persistent/index_data/EURUSD.csv"}

def load_close():
    out={}
    for s in WATCH:
        df=pd.read_csv(f"../persistent/stock_data/{s}.csv")
        df["date"]=pd.to_datetime(df["date"])
        df=df[df["date"]<=VISIBLE].sort_values("date").reset_index(drop=True)
        out[s]=df.set_index("date")["close"]
    return pd.DataFrame(out)

def load_macro():
    out={}
    for k,path in MACRO.items():
        df=pd.read_csv(path); df["date"]=pd.to_datetime(df["date"])
        df=df[df["date"]<=VISIBLE].sort_values("date").reset_index(drop=True)
        out[k]=df.set_index("date")["close"]
    return pd.DataFrame(out)

Y=load_close(); R=Y.pct_change(); M=load_macro()
horizon=10
fwd=Y.shift(-horizon)/Y - 1.0

def eval_factor(name, F, min_assets=8):
    icd=[]; icvals=[]
    for dt in F.index.intersection(fwd.index):
        pair=pd.concat([F.loc[dt], fwd.loc[dt]], axis=1, keys=['f','r']).dropna()
        if len(pair)>=min_assets:
            v=pair['f'].corr(pair['r'],method='spearman')
            if not np.isnan(v):
                icd.append(dt); icvals.append(v)
    if len(icvals)<100: return None
    icv=np.array(icvals)
    ic=icv.mean(); sd=icv.std(ddof=1)
    icir=ic/sd if sd>0 else 0.0     # pipeline convention mean/std
    hit=(icv>0).mean()
    cov=F.notna().sum().sum()/(F.shape[0]*F.shape[1])
    return {'ic':float(ic),'icir':float(icir),'hit':float(hit),'n_dates':len(icv),
            'coverage':float(cov),'abs_ic':abs(float(ic))}

def greedy_calc(macro_key, win):
    mr=M[macro_key].pct_change()
    out={}
    for s in WATCH:
        rs=Y[s].pct_change()
        d=pd.concat([rs,mr],axis=1,keys=['a','m']).dropna()
        out[s]=d['a'].rolling(win).cov(d['m'])/d['m'].rolling(win).var()
    return pd.DataFrame(out)

print("=== Candidate exploration @", VISIBLE.date(), "| days:", len(Y), "assets:", len(WATCH), "===")
cands={
 'eurusd_beta_60': greedy_calc('EURUSD',60),
 'usdjpy_beta_60': greedy_calc('USDJPY',60),
 'usdjpy_beta_20': greedy_calc('USDJPY',20),
 'dxy_beta_20':    greedy_calc('DXY',20),
 'dxy_beta_60':    greedy_calc('DXY',60),
 'usdcny_beta_20': greedy_calc('USDCNY',20),
}