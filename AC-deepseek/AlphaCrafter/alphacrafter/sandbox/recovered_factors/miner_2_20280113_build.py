"""miner_2 one candidate: residual downside-tail containment improvement (20d/60d)."""
import pathlib
src=pathlib.Path('scripts/miner_3_20271230_residual_downside_volume_vulnerability_60d.py').read_text()
# Retain the established visible-data loader, validation panel, and full reconstructed-library checks.
src=src.replace('"""miner_3 one candidate: residual downside-volume vulnerability (60d)."""','"""miner_2 one candidate: residual downside-tail containment improvement (20d/60d)."""')
src=src.replace("END=pd.Timestamp('2027-12-29')","END=pd.Timestamp('2028-01-12')")
start=src.index('# High score means fewer large idiosyncratic losses')
end=src.index('# Include all factors reconstructed', start)
new='''# High score means recent idiosyncratic downside severity has fallen versus its
# own medium-term baseline.  Residualization removes common cross-asset shocks.
neg=e.clip(upper=0)
lpm20=(neg.pow(2).rolling(20,min_periods=15).mean()).pow(.5)
lpm60=(neg.pow(2).rolling(60,min_periods=40).mean()).pow(.5)
f=-(lpm20/(lpm60+1e-12))
'''
src=src[:start]+new+src[end:]
src=src.replace("residual_downside_volume_vulnerability_60d","residual_downside_tail_containment_improvement_20_60d")
src=src.replace("if h==5:","if h==20:")
pathlib.Path('scripts/miner_2_20280113_residual_downside_tail_containment_improvement_20_60d.py').write_text(src)
print('written')
