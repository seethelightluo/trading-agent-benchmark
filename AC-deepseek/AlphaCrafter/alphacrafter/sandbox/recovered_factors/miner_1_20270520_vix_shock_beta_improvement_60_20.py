"""miner_1: one idea -- VIX shock beta improvement, 60 versus 20 sessions."""
import pathlib
src=pathlib.Path('scripts/miner_3_20270422_residual_upside_market_down_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-04-22')", "END=pd.Timestamp('2027-05-20')")
# Preserve common data/library construction; add the most recently admitted breadth signal.
posbreadth=(r>0).mean(axis=1).diff().clip(lower=0)
lib_insert="""\n# latest admitted breadth-recovery factor (60 sessions)\nbr_e=r-beta(r,m,60,40).mul(m,axis=0)\nlib['miner_1_breadth_recovery_capture_60d']=pd.DataFrame({a:br_e[a].rolling(60,min_periods=40).cov(posbreadth)/posbreadth.rolling(60,min_periods=40).var() for a in A})\n"""
src=src.replace("print('FACTOR residual_upside_in_market_down_60d'",lib_insert+"\nprint('FACTOR vix_shock_beta_improvement_60_20'")
# replace just candidate calculation beginning explanation through print as existing validation relies f
old="""# Mean positive idiosyncratic return only on broad-market down sessions, normalized by residual risk.\n# High values identify assets that retain independent upside when the cross-asset tape declines.\ndown=e.where(m<0, np.nan).clip(lower=0)\nf=down.rolling(60,min_periods=12).mean()/e.rolling(60,min_periods=40).std()\n"""
new="""# High score: short-window sensitivity to VIX changes has fallen versus its 60-session baseline.\n# This identifies assets becoming relatively more resilient to fresh volatility shocks.\nvix_level=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float)\nvix_r=vix_level.pct_change()\nf=beta(r,vix_r,60,30)-beta(r,vix_r,20,12)\n"""
src=src.replace(old,new)
# original VIX library line definition remains legal though causes variable VIX not issue
src=src.replace("src=", "src=",1)
pathlib.Path('scripts/miner_1_20270520_vix_shock_beta_improvement_60_20.py').write_text(src)
print('wrote candidate script')
