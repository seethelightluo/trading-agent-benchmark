import pandas as pd

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
R={}
for a in assets:
    df=pd.read_csv('../persistent/stock_data/%s.csv'%a)
    df['date']=pd.to_datetime(df['date'])
    R[a]=df.set_index('date')['close'].astype(float)
P=pd.DataFrame(R).sort_index()
P=P[P.index<='2032-05-14']
ret=P.pct_change()
mkt=ret.mean(axis=1)
print('rows',len(P),'last',P.index[-1].date())
print('20d mkt mean:', round(float(mkt.tail(20).mean()),5))
print('60d mkt mean:', round(float(mkt.tail(60).mean()),5))
last_close=P.iloc[-1]; ma20=P.rolling(20).mean().iloc[-1]
above=[a for a in assets if last_close[a]>ma20[a]]
below=[a for a in assets if last_close[a]<=ma20[a]]
print('ABOVE MA20:', above)
print('BELOW MA20:', below)
perf=(P.iloc[-1]/P.iloc[-26]-1).sort_values(ascending=False)
for a in perf.index:
    print('  20d',a,round(float(perf[a])*100,1),'%')
# longer perf
for name,win in [('60d',60),('120d',120)]:
    pr=(P.iloc[-1]/P.iloc[-win-1]-1).sort_values(ascending=False)
    print(name, {a:round(float(pr[a])*100,1) for a in pr.index})