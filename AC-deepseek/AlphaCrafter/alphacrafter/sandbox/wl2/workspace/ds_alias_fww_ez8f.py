lines = open('memory.txt', encoding='utf-8').readlines()
print("total lines:", len(lines))
for i, ln in enumerate(lines[-12:], start=len(lines)-11):
    print(i, ln[:300].replace("\n",""))
