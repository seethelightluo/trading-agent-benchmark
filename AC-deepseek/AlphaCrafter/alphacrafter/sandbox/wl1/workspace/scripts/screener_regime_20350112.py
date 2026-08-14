import pandas as pd, numpy as np

CUT = '2035-01-11'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = {'DXY':'DXY','USDCNY':'USDCNY','USDJPY':'USDJPY','EURUSD':'EURUSD','VIX':'VIX'}

def load(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[df['date'] <= pd.Timestamp(CUT)].reset_index(drop=True)
    return df

px = {}
for a in ASSETS:
    df = load(f'../persistent/stock_data/{a}.csv')
    px[a] = df.set_index('date')['close']

mx = {}
for k,v in MACRO.items():
    df = load(f'../persistent/index_data/{v}.csv')
    mx[k] = df.set_index('date')['close']

rets = pd.DataFrame(px)
dret = rets.pct_change().dropna()

rows = []
for a in ASSETS:
    s = rets[a]
    r = dret[a]
    last = s.iloc[-1]
    def cum(n):
        return s.iloc[-1]/s.iloc[-1-n]-1 if len(s)>n else np.nan
    ma20 = s.rolling(20).mean().iloc[-1]
    ma60 = s.rolling(60).mean().iloc[-1]
    vol20 = r.tail(20).std()*np.sqrt(252)
    vol60 = r.tail(60).std()*np.sqrt(252)
    rows.append(dict(asset=a, last=last, r20=cum(20), r60=cum(60), r120=cum(120),
                     above_ma20=1 if last>ma20 else 0, above_ma60=1 if last>ma60 else 0,
                     vol20=vol20, vol60=vol60))
tab = pd.DataFrame(rows).set_index('asset')
print("=== ASSET PANEL (through %s) ===" % CUT)
print(tab.round(4).to_string())
print()
print("EqW 20d cum: %.4f | 60d: %.4f | 120d: %.4f" % (tab.r20.mean(), tab.r60.mean(), tab.r120.mean()))
print("Breadth above MA20: %d/15 | above MA60: %d/15" % (tab.above_ma20.sum(), tab.above_ma60.sum()))
print("Mean 20d ann vol: %.2f%% | median: %.2f%%" % (tab.vol20.mean()*100, tab.vol20.median()*100))
cs_disp = dret.std(axis=1)
print("20d mean daily x-sect dispersion: %.4f%% | 60d: %.4f%% | 5d: %.4f%%" % (cs_disp.tail(20).mean()*100, cs_disp.tail(60).mean()*100, cs_disp.tail(5).mean()*100))
print()
print("=== MACRO (through %s) ===" % CUT)
for k,v in mx.items():
    last = v.iloc[-1]
    d20 = v.iloc[-1]/v.iloc[-21]-1 if len(v)>21 else np.nan
    d60 = v.iloc[-1]/v.iloc[-61]-1 if len(v)>61 else np.nan
    print(f"{k}: last {last:.2f} | 20d {d20*100:+.2f}% | 60d {d60*100:+.2f}%")
print()
print("VIX now: %.1f | 10d ago: %.1f | 20d ago: %.1f | 60d ago: %.1f" % (
    mx['VIX'].iloc[-1], mx['VIX'].iloc[-11] if len(mx['VIX'])>11 else np.nan,
    mx['VIX'].iloc[-21] if len(mx['VIX'])>21 else np.nan,
    mx['VIX'].iloc[-61] if len(mx['VIX'])>61 else np.nan))
print()
print("=== 20d RANKING ===")
print(tab.sort_values('r20')[['r20','r60','vol20']].round(4).to_string())
print()
print("=== 60d RANKING ===")
print(tab.sort_values('r60')[['r20','r60','vol20']].round(4).to_string())
