'''
time python parallel.py
real	0m15.571s
user	0m18.340s
sys	0m21.330s

Almost 1.6 times fast
'''
import multiprocessing as mp

def square(i) :
    print("num : {} ans : {}\n".format(i,i*i))
    

if __name__ == '__main__':
    nums = [i for i in range(2000000)]
    with mp.Pool(8) as p:
        p.map(square,nums)