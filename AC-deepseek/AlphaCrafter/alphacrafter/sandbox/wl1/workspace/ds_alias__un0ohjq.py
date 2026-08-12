from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
import pandas as pd

acc = get_account_dict()
nav_end = acc['net_assets']
# estimate executed cost-basis values from current pnl
pos = {p['symbol']: p for p in acc['positions']}
total_cost = sum(p['market_value']/(1+p.get('profit_loss_rate',0)/100) for p in acc['positions'])
print('est total cost (NAV @ exec):', round(total_cost,2))
print(f"{'asset':10s} {'exW%':>6s} {'blockRet%':>9s} {'contrib pp':>10s}")
rets = {'000300.SH':1.20,'SPX':0.36,'HSI':0.00,'N225':2.52,'SX5E':-4.75,'000688.SH':-5.20,'SOX':-7.71,'NDX':-5.26,'XAU':-5.13,'COPPER':4.18,'WTI':12.19,'BTC':4.33,'ETH':-21.95,'US10Y':-6.27,'CN10Y':0.00}
tot = 0
for a, r in rets.items():
    p = pos.get(a)
    if p is None: continue
    cost = p['market_value']/(1+p.get('profit_loss_rate',0)/100)
    w = cost/total_cost
    c = w*r
    tot += c
    print(f"{a:10s} {100*w:6.2f} {r:9.2f} {c:10.3f}")
print('sum contrib:', round(tot,3), 'vs actual block return -2.33%')
