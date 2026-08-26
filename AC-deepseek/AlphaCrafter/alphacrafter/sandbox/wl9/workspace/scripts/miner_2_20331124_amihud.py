import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_stock_daily_data

watch = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DAYS = 2500

# Load close and volume panels, align on dates
closes = {}
vols = {}
for s in watch:
    df = get_stock_daily_data(symbol=s, days=DAYS)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    closes[s] = df['close']
    vols[s] = df['volume']
close_df = pd.DataFrame(closes)
vol_df = pd.DataFrame(vols)
print('panel dates', close_df.shape)

ret = close_df.pct_change()

# Factor: Amihud illiquidity 20d = mean(|ret| / volume) over rolling 20 days
ill = ret.abs() / (vol_df + 1e-12)
amihud = ill.rolling(20).mean().shift(1)  # shift to avoid look-ahead
amihud_log = np.log(amihud + 1e-12)

def compute_ic(factor, forward_horizon=10):
    fwd = close_df.pct_change(forward_horizon).shift(-forward_horizon)
    ic_series = []
    dates_ge8 = []
    for dt, row in factor.iterrows():
        f = row.dropna()
        fw = fwd.loc[dt]
        fw = fw.dropna()
        common = f.index.intersection(fw.index)
        if len(common) >= 8:
            ic = np.corrcoef(f[common], fw[common])[0,1]
            if np.isfinite(ic):
                ic_series.append((dt, ic, len(common)))
                dates_ge8.append(dt)
    return ic_series, dates_ge8

for name, fac in [('amihud', amihud_log)]:
    ic_series, dates8 = compute_ic(fac, 10)
    ics = np.array([x[1] for x in ic_series])
    print(f'=== {name} horizon10 ===')
    print('n dates', len(ics), 'n dates ge8', len(dates8))
    print('mean IC', round(ics.mean(),4), 'std', round(ics.std(),4), 'ICIR', round(ics.mean()/ics.std(),4))
    print('IC hit ratio', round((ics>0).mean(),4))
    # coverage
    cov_days = fac.notna().sum(axis=1)
    print('coverage asset-days mean', round(cov_days.mean()/15,4))
    # decay
    for h in [1,2,3,5,10,20]:
        ic, _ = compute_ic(fac, h)
        iarr = np.array([x[1] for x in ic])
        print(f'  h{h}: IC {round(iarr.mean(),4)} ICIR {round(iarr.mean()/iarr.std(),4)}')

# turnover: rank change in factor between consecutive dates
factor = amihud_log
ranks = factor.rank(axis=1)
turn = ranks.diff(10).abs().mean(axis=1).mean()
print('turnover 10d rank chg mean', round(turn,4))