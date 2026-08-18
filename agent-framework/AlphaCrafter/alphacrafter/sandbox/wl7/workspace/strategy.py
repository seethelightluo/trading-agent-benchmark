import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights
UNIVERSE=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
ENSEMBLE_PATH=Path(__file__).parent/"factors"/"factor_ensemble.json"
_last_decision=None

def rank(v,d=1):
    z=sorted((s,d*float(x)) for s,x in v.items() if np.isfinite(x)); o={s:.5 for s in UNIVERSE}
    for i,(s,_) in enumerate(z): o[s]=(i+1)/max(len(z),1)
    return o

def bounded(raw):
    w={s:max(float(raw.get(s,.01)),1e-12) for s in UNIVERSE}; fixed=set()
    for _ in range(30):
        free=[s for s in UNIVERSE if s not in fixed]; scale=(1-sum(w[s] for s in fixed))/max(sum(w[s] for s in free),1e-12); changed=False
        for s in free:
            w[s]*=scale
            if w[s]<.015: w[s]=.015; fixed.add(s); changed=True
            elif w[s]>.16: w[s]=.16; fixed.add(s); changed=True
        if not changed: break
    t=sum(w.values()); return {s:w[s]/t for s in UNIVERSE}

def load():
    try:
        a=json.loads(ENSEMBLE_PATH.read_text()); q=[x for x in a.get('selected_factors',[]) if x.get('factor_id')][:10]
        return q,{x['factor_id']:float(x.get('weight',0)) for x in q},{x['factor_id']:int(x.get('direction',1)) for x in q}
    except Exception: return [],{},{}

@register_hook
def cross_asset_strategy():
    global _last_decision
    selected,fw,d=load()
    if not selected:return
    data={}
    for s in UNIVERSE:
        df=get_stock_daily_data(symbol=s,days=90)
        if df is None or len(df)<25: continue
        df=df.sort_values('date').reset_index(drop=True); c=np.asarray(df.close,float); h=np.asarray(df.high,float); l=np.asarray(df.low,float); r=c[1:]/np.maximum(c[:-1],1e-12)-1
        data[s]=(c,h,l,r,str(df.iloc[-1].date))
    if len(data)<12:return
    date=max(v[4] for v in data.values())
    if _last_decision is not None and (np.datetime64(date)-np.datetime64(_last_decision))/np.timedelta64(1,'D')<10:return
    f={x:{} for x in fw}; iv={}; r5={}
    for s,(c,h,l,r,_) in data.items():
        vol=max(float(np.std(r[-20:])),.008); iv[s]=1/vol; r5[s]=c[-1]/max(c[-6],1e-12)-1
        vals={'miner_3_clv_1d':(2*c[-1]-h[-1]-l[-1])/max(h[-1]-l[-1],1e-12),'short_term_reversal_5d':-np.mean(r[-5:]),'miner_3_intraday_reversal_1d':-r[-1],'miner_2_risk_adjusted_momentum_20d':(c[-1]/max(c[-21],1e-12)-1)/(vol+.01)}
        for k,v in vals.items():
            if k in f:f[k][s]=v
    if 'peer_median_leadlag_5d' in f:
        m=float(np.median(list(r5.values()))); f['peer_median_leadlag_5d']={s:v-m for s,v in r5.items()}
    rr={k:rank(v,d.get(k,1)) for k,v in f.items()}; score={s:sum(fw[k]*rr[k].get(s,.5) for k in fw) for s in UNIVERSE}
    if 'SPX' in data and data['SPX'][0][-1]<data['SPX'][0][-21]:
        for s in ('XAU','US10Y','CN10Y'):score[s]+=.10
        for s in ('BTC','ETH','WTI'):score[s]=max(.01,score[s]-.06)
    avg=np.mean(list(iv.values())); raw={s:max(score[s],.01)*(.78+.22*iv.get(s,avg)/max(avg,1e-12)) for s in UNIVERSE}; w=bounded(raw)
    mu=np.mean(list(score.values())); sd=max(np.std(list(score.values())),1e-12); forecast={s:float(.01*(score[s]-mu)/sd) for s in UNIVERSE}
    rebalance_to_weights(w,forecast_returns=forecast,factor_ids=[x['factor_id'] for x in selected],horizon_days=10); _last_decision=date
