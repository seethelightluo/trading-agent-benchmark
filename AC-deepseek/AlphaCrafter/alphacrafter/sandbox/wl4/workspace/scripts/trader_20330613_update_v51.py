from pathlib import Path

p = Path("strategy.py")
s = p.read_text()

v51 = '"""v51 (2033-06-13):\n'
v51 += 'v50 + three risk-adjustment fires from the 05-30..06-13 block (proposal on\n'
v51 += '05-30 EXECUTED, cost = 05-27 closes; account 1153530.85 -> 1168482.90,\n'
v51 += '+1.30% block, Sharpe 8.28 DD 0.29%; SCREENER ensemble 0.50/0.25/0.25\n'
v51 += 'unchanged since 05-02, loaded live from factors/factor_ensemble.json):\n'
v51 += '  - NDX x0.75 -> x0.60  3rd consecutive negative block (-1.68% 05-02..05-16,\n'
v51 += '    -2.55% 05-16..05-30, -4.75% 05-30..06-13, cumulative ~-8.7%, MAIN DRAG\n'
v51 += '    ~-0.38% this block at w~0.08); fires the v50 watch "NDX x0.60 on 2nd\n'
v51 += '    cons neg or <-6%" (chain-cut mirror SOX/US10Y/000688/N225 patterns).\n'
v51 += '  - SX5E x0.60 -> x0.70  3rd consecutive positive block (+2.25%, +1.47%,\n'
v51 += '    +7.96% TOP winner this block at w~0.045); fires the v50 watch "SX5E\n'
v51 += '    x0.70 on 3rd cons pos".\n'
v51 += '  - 000688.SH x0.60 -> x0.70  3rd consecutive positive block (+1.25%,\n'
v51 += '    +21.41%, +0.12%); fires the v50 watch "000688 x0.70 on 2 cons pos".\n'
v51 += '  Kept: WTI x0.15 (STRONG pos +17.37% 1st after the 4-cons-neg air-pocket\n'
v51 += '  chain, contained at w~0.016; re-boost x0.20 on 2 stable), SOX x0.20\n'
v51 += '  (STRONG pos +9.83% 1st after the -10.47%/-1.08% negs; re-boost x0.30 on\n'
v51 += '  2 cons pos), N225 x0.60 (1st mild neg -1.05% after 2 cons pos; re-boost\n'
v51 += '  x0.70 on 2 cons pos), US10Y x0.30 (1st neg -0.83% after 2 cons pos;\n'
v51 += '  re-boost x0.35 on 2 cons pos), XAU x1.00 (3rd cons pos +3.26% at cap\n'
v51 += '  w~0.076; x0.85 on 2nd cons neg or <-5%), SPX x0.75 (3rd cons pos +0.95%;\n'
v51 += '  x0.85 on 2 more cons pos), COPPER cap 0.12 (pos +4.05% w~0.115; x0.85 on\n'
v51 += '  2nd cons neg or <-8%), CN10Y x0.70/ETH x0.75/BTC x0.15 frozen stale.\n'
v51 += '  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~37% book\n'
v51 += '  neutral rank, pnl 0).\n'
v51 += '  Ensemble: 0.50/0.25/0.25 (SCREENER 2033-05-02 re-tilt) - loaded live\n'
v51 += '  from factors/factor_ensemble.json.\n'
v51 += '"""\n\n\n'

old = '"""v50 (2033-05-16):'
assert old in s, "v50 header not found"
s = s.replace(old, v51 + old, 1)
p.write_text(s)
print("v51 docstring inserted OK")
