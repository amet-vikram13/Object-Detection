'''
time python parallel.py
real	1m23.973s
user	0m42.962s
sys	0m53.176s
'''
import multiprocessing as mp
from multiprocessing import Manager,Value

## global variable
nums = Manager().list(range(2000000))
#ans  = Manager().list([])
x = Value('i',0)
ans = 0

def square(nums,id) :
    global x
    global ans
    for i in range(len(nums)) :
        x.value += 1
        #ans.append((nums[i],nums[i]*nums[i]))
        ans = nums[i]*nums[i]
        #print(x.value)
        #print("num : {} ans : {}\n".format(nums[i],nums[i]*nums[i]))
    print("process {} done!\n".format(id))
    
chunk = int(len(nums)/2)

procs = []

for i in range(2) :
    p = mp.Process(target=square,args=(nums[chunk*i:chunk*(i+1)],i))
    procs.append(p)
    p.start()

for p in procs :
    p.join()

print(x.value)
#print(len(ans))