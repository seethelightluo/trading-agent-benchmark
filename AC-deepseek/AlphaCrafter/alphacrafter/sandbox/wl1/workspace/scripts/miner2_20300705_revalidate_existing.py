"""miner2 2030-07-05: re-validate existing miner2 factor library on fresh panel (through 2030-07-04)."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')
from miner2_val_lib import eval_factor, summarize, daily_rank_ic, fwd_ret

P = pd.read_pickle('scripts/panel_cache_20300705.pkl')
close, high, low, open_, ret = P['close'], P['high'], P['low'], P['open'], P['ret']

factors = {}
# short-term reversal family
for nd in [1, 2, 3, 5]:
    factors[f'rev_{nd}d'] = -(np.log(close) - np.log(close.shift(nd)))
# negative close location value family
for nd in [1, 2, 3, 5]:
    rng = high.rolling(nd).max() - low.rolling(nd).min()
    factors[f'nclv_{nd}d'] = -(close - low.rolling(nd).min()) / rng.replace(0, np.nan)
# intraday reversal
factors['id_rev_1d'] = -(close / open_ - 1.0)
# negative body
factors['nbody_1d'] = -(close - open_) / (high - low).replace(0, np.nan)
# vol-scaled reversal 1d
factors['rev_1d_vs'] = -(np.log(close) - np.log(close.shift(1))) / ret.rolling(20).std().replace(0, np.nan)

FULL_START, FULL_END = '2021-01-01', '2030-07-04'
RECENT_START = '2028-01-01'
RECENT2_START = '2029-06-01'

print("=" * 112)
print(f"RE-VALIDATION of existing miner2 factors | FULL {FULL_START}..{FULL_END} | RECENT {RECENT_START}..{FULL_END}")
print("=" * 112)
full_res, recent_res, recent2_res = {}, {}, {}
for name, sig in factors.items():
    r_full = eval_factor(sig, close, start=FULL_START, end=FULL_END)
    r_recent = eval_factor(sig, close, start=RECENT_START, end=FULL_END)
    r_recent2 = eval_factor(sig, close, start=RECENT2_START, end=FULL_END)
    full_res[name] = r_full
    recent_res[name] = r_recent
    recent2_res[name] = r_recent2
    summarize(r_full, f"FULL  {name}")
    summarize(r_recent, f"RECENT {name}")
    summarize(r_recent2, f"R2029+ {name}")

# year-by-year IC1
print("\nYear-by-year IC1 (full sample):")
for name in ['rev_1d', 'rev_2d', 'nclv_1d', 'nclv_2d', 'id_rev_1d', 'nbody_1d']:
    sig = factors[name]
    yrs = {}
    for y in range(2021, 2031):
        r = eval_factor(sig, close, start=f'{y}-01-01', end=f'{y}-12-31')
        yrs[str(y)] = r.get(1, {})
        h1 = r.get(1, {})
        print(f"  {name} {y}: IC1={h1.get('ic', float('nan')):.4f} ICIR1={h1.get('icir', float('nan')):.3f} n={h1.get('n', 0)}")
    full_res[name]['by_year_ic1'] = yrs

with open('scripts/miner2_reval_20300705.json', 'w') as f:
    json.dump({'full': {k: {str(ik): iv for ik, iv in v.items()} for k, v in full_res.items()},
               'recent': {k: {str(ik): iv for ik, iv in v.items()} for k, v in recent_res.items()},
               'recent2': {k: {str(ik): iv for ik, iv in v.items()} for k, v in recent2_res.items()}}, f, indent=1, default=str)
print("\nsaved scripts/miner2_reval_20300705.json")
