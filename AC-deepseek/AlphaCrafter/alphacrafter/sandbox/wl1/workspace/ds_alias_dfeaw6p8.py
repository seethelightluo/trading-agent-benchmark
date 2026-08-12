from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

acc = get_account_dict()
pos = {p['symbol']: p for p in acc['positions']}
nav_exec = 959030.99  # pre-step NAV @12-20 (cash=0 fully invested)
px = {}
for a in pos:
    df = get_stock_daily_data(symbol=a, days=30)
    df = df.copy(); df['date'] = pd.to_datetime(df['date']); df = df.set_index('date').sort_index()
    d0 = df.index[df.index <= pd.Timestamp('2030-12-19')]
    px[a] = float(df.loc[d0[-1], 'close'])
print(f"{'asset':10s} {'qty':>12s} {'px@12-19':>10s} {'exW%':>6s}")
tot = 0
w = {}
for a, p in pos.items():
    q = p['quantity']
    v = q * px[a]
    w[a] = v
    tot += v
for a, v in sorted(w.items(), key=lambda x: -x[1]):
    print(f"{a:10s} {pos[a]['quantity']:12.4f} {px[a]:10.4f} {100*v/tot:6.2f}")
print('total exec value:', round(tot,2), 'vs NAV 959,031')
btc_eth = 100*(w.get('BTC',0)+w.get('ETH',0))/tot
wti_cop = 100*(w.get('WTI',0)+w.get('COPPER',0))/tot
print('BTC+ETH combined %:', round(btc_eth,2), ' WTI+COPPER combined %:', round(wti_cop,2))
