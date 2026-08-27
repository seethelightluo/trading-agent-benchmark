# Check the latest market state - VIX and other macro indicators
from alphacrafter.sim.utils import get_index_daily_data
import pandas as pd

# Get VIX data
vix = get_index_daily_data("VIX", 250)
if vix is not None:
    print("VIX last 10 days:")
    print(vix[['date', 'close']].tail(10).to_string())
    print(f"\nVIX current: {vix['close'].iloc[-1]:.2f}")
    print(f"VIX 20d ago: {vix['close'].iloc[-21]:.2f}")
    print(f"VIX 60d ago: {vix['close'].iloc[-61]:.2f}")
    vix_roc_20 = vix['close'].iloc[-1] / vix['close'].iloc[-21] - 1
    vix_roc_60 = vix['close'].iloc[-1] / vix['close'].iloc[-61] - 1
    print(f"VIX 20d ROC: {vix_roc_20:.4f}")
    print(f"VIX 60d ROC: {vix_roc_60:.4f}")
    # Acceleration
    vix_roc_20_prior = vix['close'].iloc[-21] / vix['close'].iloc[-41] - 1
    accel = vix_roc_20 - vix_roc_20_prior
    print(f"VIX 20d ROC 20d ago: {vix_roc_20_prior:.4f}")
    print(f"VIX acceleration (20d ROC change): {accel:.4f}")
    
    # Also mean reversion metrics
    vix_mean_60 = vix['close'].tail(60).mean()
    vix_median_60 = vix['close'].tail(60).median()
    vix_z = (vix['close'].iloc[-1] - vix_mean_60) / vix['close'].tail(60).std()
    print(f"VIX 60d mean: {vix_mean_60:.2f}, median: {vix_median_60:.2f}, z-score: {vix_z:.2f}")
    print(f"VIX min 60d: {vix['close'].tail(60).min():.2f}, max: {vix['close'].tail(60).max():.2f}")
else:
    print("No VIX data")