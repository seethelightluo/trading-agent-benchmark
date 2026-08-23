import glob
syms=['SPX','XAU','WTI','BTC','ETH','N225','HSI','SOX','SX5E','NDX','000300.SH','US10Y','CN10Y','COPPER','000688.SH']
for s in syms:
    try:
        lines=[l.strip().split(',') for l in open(f'../persistent/stock_data/{s}.csv') if l.strip()]
        closes=[l[1] for l in lines if l[1].replace('.','').replace('-','').isdigit()]
        # header may be first row
        closes=[float(x) for x in closes]
        r=(closes[-1]/closes[-60]-1)*100
        print('%-9s 60d_chg %6.1f%%  close %9.2f'%(s,r,closes[-1]))
    except Exception as e:
        print(s,'ERR',e)