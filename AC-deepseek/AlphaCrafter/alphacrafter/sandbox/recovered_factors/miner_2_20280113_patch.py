import pathlib
p=pathlib.Path('scripts/miner_2_20280113_residual_downside_tail_containment_improvement_20_60d.py')
s=p.read_text().replace("neg=e.clip(upper=0)\nlpm20", "neg=e.clip(upper=0)\nlv=np.log(vol.replace(0,np.nan)); vs=lv-lv.rolling(20,min_periods=15).mean()\nlpm20")
p.write_text(s)
