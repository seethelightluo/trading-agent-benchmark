lines = open('memory.txt').read().splitlines()
# print last 8 lines fully
for l in lines[-8:]:
    print(l)
    print('==========')
