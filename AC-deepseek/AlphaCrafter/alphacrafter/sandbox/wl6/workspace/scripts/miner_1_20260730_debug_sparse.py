"""Debug: why do vol_of_vol / vix_beta signals look sparse vs persisted metrics."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner_1_20260730_validation_lib import load_close_panel, load_macro_panel

panel = load_close_panel()
print('panel shape:', panel.shape)
print('per-asset non-NaN count:')
print(panel.notna().sum())
print('panel head dates:', panel.index[:3].date, 'tail:', panel.index[-3:].date)

ret = panel.pct_change()
print('\nret non-NaN per asset:', ret.notna().sum().tolist())
v1 = ret.rolling(20).std()
print('vol20 non-NaN per asset:', v1.notna().sum().tolist())
v2 = v1.rolling(60).std()
print('vol_of_vol non-NaN per asset:', v2.notna().sum().tolist())
print('vol_of_vol total non-NaN cells:', int(v2.notna().sum().sum()))

macro = load_macro_panel()
print('\nmacro panel shape:', macro.shape)
vixr = macro['VIX'].pct_change()
beta = ret.rolling(60).cov(vixr) / vixr.rolling(60).var()
print('beta non-NaN per asset:', beta.notna().sum().tolist())
vixchg = macro['VIX'] / macro['VIX'].shift(20) - 1.0
sig = -beta * vixchg
print('vix_beta sig non-NaN per asset:', sig.notna().sum().tolist())
