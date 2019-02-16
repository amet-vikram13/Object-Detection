import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

image_path       = "images/"
train_org_df     = pd.read_csv("training.csv")

train_org_name   = train_org_df["image_name"].values

image_matrix = np.arange(1000*480*640*3).reshape(1000,480,640,3)
image_matrix.fill(0)

j = 0

for i in range(len(train_org_name)+1) :
	if i%1000==0 and i!=0 :
		np.savez_compressed('./image_arrays/arrays_+'+str(j)+'_'+str(i-1),image_matrix=image_matrix)
		print("############BATCH DONE###############")
		image_matrix.fill(0)
		j = i
	if i==14000 :
		print("####DONE#####")
		break
	img = plt.imread(image_path+train_org_name[i])
	image_matrix[i,:,:,:] = img
	print("{} done!".format(i))
