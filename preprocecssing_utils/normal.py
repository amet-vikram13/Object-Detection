'''
time python normal.py
real	0m23.992s
user	0m11.969s
sys	0m6.896s
'''
def square(i) :
    return i*i

x = 0
ans = 0
for i in range(2000000) :
    x += 1
    ans = square(i)
    print(x)
    #print("num : {} ans : {}\n".format(i,square(i)))
