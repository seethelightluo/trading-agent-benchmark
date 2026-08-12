"""miner2 2030-03-29: re-validate existing miner2 factor library on fresh data."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')
from miner2_val_lib import eval_factor, summarize

P = pd.read_pickle('scripts/panel_cache_20300329.pkl')
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

FULL_START, FULL_END = '2021-01-01', '2030-03-28'
RECENT_START = '2028-01-01'

print("=" * 110)
print(f"RE-VALIDATION of existing miner2 factors | FULL {FULL_START}..{FULL_END} | RECENT {RECENT_START}..{FULL_END}")
print("=" * 110)
full_res, recent_res = {}, {}
for name, sig in factors.items():
    r_full = eval_factor(sig, close, start=FULL_START, end=FULL_END)
    r_recent = eval_factor(sig, close, start=RECENT_START, end=FULL_END)
    full_res[name] = r_full
    recent_res[name] = r_recent
    summarize(r_full, f"FULL  {name}")
    summarize(r_recent, f"RECENT {name}")

with open('scripts/miner2_reval_20300329.json', 'w') as f:
    json.dump({'full': {k: {str(ik): iv for ik, iv in v.items()} for k, v in full_res.items()},
               'recent': {k: {str(ik): iv for ik, iv in v.items()} for k, v in recent_res.items()}}, f, indent=1, default=str)
print("saved scripts/miner2_reval_20300329.json")
