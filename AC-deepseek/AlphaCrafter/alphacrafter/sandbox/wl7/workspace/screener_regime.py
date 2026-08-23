import pandas as pd

for s in ['VIX','DXY','EURUSD','USDJPY','USDCNY']:
    df=pd.read_csv('../persistent/index_data/%s.csv'%s)
    df['date']=pd.to_datetime(df['date'])
    df=df[df['date']<='2032-05-14']
    last=df['close'].iloc[-1]
    m20=last/df.set_index('date')['close'].iloc[-21]-1
    m60=last/df.set_index('date')['close'].iloc[-61]-1
    print(s,'last',df['date'].iloc[-1].date(),'close',round(float(last),2),
          '20d',round(float(m20)*100,1),'%','60d',round(float(m60)*100,1),'%')

# per-asset 60d vol (annualized)
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
R={}
for a in assets:
    df=pd.read_csv('../persistent/stock_data/%s.csv'%a)
    df['date']=pd.to_datetime(df['date'])
    R[a]=df.set_index('date')['close'].astype(float)
P=pd.DataFrame(R).sort_index(); P=P[P.index<='2032-05-14']
ret=P.pct_change()
vol=ret.tail(30).std()*(252**0.5)
print('median 30d vol ann:', round(float(vol.median())*100,1),'%')
for a in vol.sort_values(ascending=False).index:
    print('  ',a,round(float(vol[a])*100,1),'%')
# frozen flags: std near zero
std=ret.tail(120).std()
frozen=[a for a in assets if std[a]<1e-9]
print('frozen (zero 120d vol):', frozen)
print('dispersion: std of 20d per-asset returns:', round(float((P.iloc[-1]/P.iloc[-26]-1).std())*100,2),'%')