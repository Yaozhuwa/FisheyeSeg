x = [['1','1'],['2','2']]
def f(x):
    return list(map(float,x))

print(list(map(f,x)))
# print(float(x))