import datetime
cut='2033-09-13'
syms=['SPX','XAU','WTI','BTC','ETH','N225','HSI','SOX','SX5E','NDX','000300.SH','US10Y','CN10Y','COPPER','000688.SH']
for s in syms:
    try:
        rows=[l.strip().split(',') for l in open(f'../persistent/stock_data/{s}.csv') if l.strip()]
        data=[(r[0],float(r[1])) for r in rows[1:] if r[0]<=cut and r[1].replace('.','').isdigit()]
        c=[x[1] for x in data]
        r60=(c[-1]/c[-61]-1)*100
        r20=(c[-1]/c[-21]-1)*100
        # 10d vol of daily returns (last ~20)
        rets=[c[i]/c[i-1]-1 for i in range(1,len(c))]
        v10=(__import__('statistics').pstdev(rets[-20:])*(252**0.5))*100
        print('%-9s 20d %6.1f%% 60d %6.1f%%  ann_vol20d %5.1f%%  close %9.2f'%(s,r20,r60,v10,c[-1]))
    except Exception as e:
        print(s,'ERR',e)