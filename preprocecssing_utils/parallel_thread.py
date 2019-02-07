'''
time python parallel_thread.py
real	1m2.757s
user	0m44.485s
sys	0m53.872s
'''
import threading as th

## global variable
nums = [i for i in range(2000000)]
x = 0

def square(nums) :
    global x
    for i in range(len(nums)) :
        x += 1
        print("num : {} ans : {}\n".format(nums[i],nums[i]*nums[i]))
    
chunk = int(len(nums)/8)
thread_list = []

for i in range(8) :
    t = th.Thread(target=square,args=(nums[chunk*i:chunk*(i+1)],))
    thread_list.append(t)
    t.start()

for t in thread_list :
    t.join()

print(x)