'''
We need to implement this file using any processing 
module like multiprocessing,threadimg or concurrent.futures
in Jagsi's laptop because thats our best shot to get all 14000
images arrays and their corresponding coordinates.
In my laptop its taking :

amet@inspiron-7520:~/GithubProjects/objectDetection$ time python optimization.py 

real	0m5.107s
user	0m2.257s
sys	0m0.211s

5 seconds to process 10 images so for 14000 images it will roughly take 1.94 hours
Also the final arrays.npz file is of size 4.7MB so final size of all arrays will be
roughly 6.58 GB

We can reduce this stats a lot if optimization implemented properly


Using multiprocessing.Pool refer normal.py and parallel.py
'''

import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

image_path       = "images/"
train_org_df     = pd.read_csv("training.csv")

train_org_df     = train_org_df[:20]  ## Testing for only 10 images, we have to do for 14000 images

train_org_name   = train_org_df["image_name"].values
coord            = train_org_df.drop(['image_name'],axis=1).values


matrix_coord = []

'''
for i in range(len(train_org_name)) :
    img = plt.imread(image_path+train_org_name[i])
    matrix_coord.append([img,coord[i]])
'''
def image_matrix(i) :
    global image_path
    global train_org_name
    global coord
    global matrix_coord
    img = plt.imread(image_path+train_org_name[i])
    print(img[1,1,:],"\n")
    matrix_coord.append([img[1,1,:],coord[i]])
    print(len(matrix_coord))


with mp.Pool(8) as p :
    p.map(image_matrix,[i for i in range(20)])

print(matrix_coord)

final_arr = np.array(matrix_coord)

print(final_arr)

np.savez_compressed('./arrays',train_org_arr=final_arr)

