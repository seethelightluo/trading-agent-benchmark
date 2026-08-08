from pathlib import Path
p=Path('scripts/miner_1_20280210_residualized_crypto_basket_correlation_decoupling_60_20.py')
s=p.read_text().replace('base.replace("END=pd.Timestamp(\'2028-02-09\')","END=pd.Timestamp(\'2028-02-09\')")', 'base.replace("END=pd.Timestamp(\'2028-01-26\')","END=pd.Timestamp(\'2028-02-09\')")')
p.write_text(s)
print('patched')
